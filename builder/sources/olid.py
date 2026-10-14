# builder/sources/olid.py
"""OLID (Zampieri et al., 2019) with AbuseEval (Caselli et al., 2020): 14,100 tweets labelled offensive or not, then
targeted or not, then at an individual, a group or other, with AbuseEval's later abusive (explicit or implicit) or not
abusive label on each. A tweet is hateful when it is an offence targeted at a group that AbuseEval finds abusive,
offensive when otherwise offensive, and neither when not offensive; explicitness is AbuseEval's reading of the offence.
Only "other" targets name a category. OLID states no licence, and its official page now requires a sign-in, so its
files come from a Hugging Face mirror, checked against the paper's counts; AbuseEval's labels are CC BY-NC-SA 4.0.
OLID replaced mentions with @USER and links with URL.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.olid"""
import csv
import urllib.request
from collections import Counter
from pathlib import Path

from builder.schema import Record

OLID = "https://huggingface.co/datasets/christophsonntag/OLID/resolve/main/data/"
ABUSEEVAL = "https://raw.githubusercontent.com/tommasoc80/AbuseEval/master/data/"
CACHE = Path("data/sources/olid")
FILES = {
    **{name: OLID + name for name in ("olid-training-v1.0.tsv", "testset-levela.tsv", "labels-levela.csv",
                                      "labels-levelb.csv", "labels-levelc.csv")},
    "abuseeval_abuseval_offenseval_train.tsv": ABUSEEVAL + "abuseval_labels/abuseval_offenseval_train.tsv",
    "abuseeval_abuseval_offenseval_test.tsv": ABUSEEVAL + "abuseval_labels/abuseval_offenseval_test.tsv",
    "abuseeval_offenseval_exp_imp_train.tsv": ABUSEEVAL + "offenseval_explicit_implicit/offenseval_exp_imp_train.tsv",
    "abuseeval_offenseval_exp_imp_test.tsv": ABUSEEVAL + "offenseval_explicit_implicit/offenseval_exp_imp_test.tsv",
}
LICENCE = "unspecified"
CITATION = ("Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N. and Kumar, R. (2019). Predicting the "
            "Type and Target of Offensive Posts in Social Media. NAACL-HLT 2019. https://aclanthology.org/N19-1144; "
            "Caselli, T., Basile, V., Mitrović, J., Kartoziya, I. and Granitzer, M. (2020). I Feel Offended, Don't "
            "Be Abusive! Implicit/Explicit Messages in Offensive and Abusive Language. LREC 2020. "
            "https://aclanthology.org/2020.lrec-1.760")
EXPLICIT = {"EXP": "explicit", "IMP": "implicit", "O": "unknown"}
TABLE_3 = {  # the OLID paper's training and test tweets per label combination
    ("OFF", "TIN", "IND"): (2407, 100), ("OFF", "TIN", "OTH"): (395, 35), ("OFF", "TIN", "GRP"): (1074, 78),
    ("OFF", "UNT", "NULL"): (524, 27), ("NOT", "NULL", "NULL"): (8840, 620),
}
ABUSE = {  # the AbuseEval paper's training and test tweets per abuse label and OLID level
    ("EXP", "TIN"): (1887, 103), ("EXP", "UNT"): (136, 3), ("IMP", "TIN"): (668, 70), ("IMP", "UNT"): (58, 2),
    ("NOTABU", "TIN"): (1321, 40), ("NOTABU", "UNT"): (330, 22), ("NOTABU", "NULL"): (8840, 620),
}


def table(path: Path, delimiter: str = "\t") -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def fetch(cache: Path = CACHE) -> list[dict]:
    """Every tweet with its three OLID levels, its split and AbuseEval's two labels, the files downloaded once."""
    for name, url in FILES.items():
        if not (cache / name).exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, cache / name)
    gold = {}
    for level in "abc":
        with (cache / f"labels-level{level}.csv").open(encoding="utf-8", newline="") as f:
            gold[level] = dict(csv.reader(f))
    tweets = [{**r, "split": "train"} for r in table(cache / "olid-training-v1.0.tsv")]
    tweets += [{"id": r["id"], "tweet": r["tweet"], "split": "test", "subtask_a": gold["a"][r["id"]],
                "subtask_b": gold["b"].get(r["id"], "NULL"), "subtask_c": gold["c"].get(r["id"], "NULL")}
               for r in table(cache / "testset-levela.tsv")]
    abuse = {r["id"]: r["abuse"] for s in ("train", "test")
             for r in table(cache / f"abuseeval_abuseval_offenseval_{s}.tsv")}
    explicit = {r["id"]: r["implicit_explicit"] for s in ("train", "test")
                for r in table(cache / f"abuseeval_offenseval_exp_imp_{s}.tsv")}
    return [{**t, "abuse": abuse[t["id"]], "explicit": explicit[t["id"]]} for t in tweets]


def level(t: dict) -> str:
    if t["subtask_a"] == "NOT":
        return "neither"
    return "hateful" if t["subtask_c"] == "GRP" and t["abuse"] != "NOTABU" else "offensive"


def records(tweets: list[dict]) -> list[Record]:
    return [Record(
        source="olid", source_id=t["id"], platform="twitter", text=t["tweet"], text_form="preprocessed", hate=level(t),
        label_raw=f"{t['subtask_a']}/{t['subtask_b']}/{t['subtask_c']} abuse={t['abuse']}",
        target_category=("other",) if t["subtask_c"] == "OTH" else (), target_group=(), n_annotators=None,
        soft_label=None, licence=LICENCE, explicitness=EXPLICIT[t["explicit"]]) for t in tweets]


def check(tweets: list[dict], out: list[Record]) -> None:
    """Fails unless the tweets reproduce the OLID paper's Table 3 and the AbuseEval paper's abuse counts, split by
    split, with each tweet once; then prints the totals."""
    def by_split(key) -> dict:
        counts = Counter((key(t), t["split"]) for t in tweets)
        return {k: (counts[k, "train"], counts[k, "test"]) for k, _ in counts}
    if (olid := by_split(lambda t: (t["subtask_a"], t["subtask_b"], t["subtask_c"]))) != TABLE_3:
        raise ValueError(f"label combinations {olid}; the OLID paper gives {TABLE_3}")
    if (abuse := by_split(lambda t: (t["abuse"], t["subtask_b"]))) != ABUSE:
        raise ValueError(f"abuse labels {abuse}; the AbuseEval paper gives {ABUSE}")
    if len({t["id"] for t in tweets}) != len(tweets):
        raise ValueError("a tweet appears twice")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    tweets = fetch()
    out = records(tweets)
    check(tweets, out)
    return out


if __name__ == "__main__":
    build()
