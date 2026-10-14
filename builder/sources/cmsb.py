# builder/sources/cmsb.py
"""\"Call me sexist, but\" (Samory et al., 2021): the 2,636 tweets its own crowdworkers annotated, five each, for sexist
content and for sexist, uncivil or civil phrasing: tweets beginning "call me sexist, but", and the benevolent and
hostile sexism tweets of Jha and Mamidi (2017) and Waseem and Hovy (2016), from GESIS after registration (CC BY-NC-SA
4.0). Psychology scale items and the adversarial rewrites are left out, as is the "other" set, whose labels are
inherited rather than annotated. Per annotation, sexist content or sexist phrasing (the authors' rule) is hateful,
uncivil but non-sexist phrasing offensive, and the rest neither; a tweet takes the strict majority, unknown without
one. Sexist tweets target gender; their explicitness is our reading of the codebook: sexist phrasing is explicit,
sexism in content alone implicit. Mentions are anonymised by the authors.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.cmsb"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from builder.schema import SOFT_LEVELS, Record

DOI = "https://doi.org/10.7802/2251"
CACHE = Path("data/sources/cmsb")
LICENCE = "CC-BY-NC-SA-4.0"
CITATION = ("Samory, M., Sen, I., Kohne, J., Flöck, F. and Wagner, C. (2021). “Call me sexist, but...”: Revisiting "
            "Sexism Detection Using Psychological Scales and Adversarial Samples. ICWSM 2021. "
            "https://doi.org/10.1609/icwsm.v15i1.18085; Samory, M. (2021). The 'Call me sexist but' Dataset (CMSB). "
            "GESIS. https://doi.org/10.7802/2251")
TWEETS = ("callme", "benevolent", "hostile")


def fetch(cache: Path = CACHE) -> tuple[list[dict], dict[str, list[dict]]]:
    """The original tweets and every annotation by tweet id, from the user's own download."""
    data, notes = cache / "sexism_data.csv", cache / "sexism_annotations.csv"
    if not (data.exists() and notes.exists()):
        raise FileNotFoundError(f"{cache}: register with GESIS and download the dataset at {DOI} into this folder")
    with data.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    annotations = defaultdict(list)
    with notes.open(encoding="utf-8", newline="") as f:
        for a in csv.DictReader(f):
            annotations[a["id"]].append(a)
    return rows, annotations


def sexist(a: dict) -> bool:
    return a["content"] in ("1", "2", "3", "4") or a["phrasing"] == "1"


def level(a: dict) -> str:
    return "hateful" if sexist(a) else "offensive" if a["phrasing"] == "2" else "neither"


def records(rows: list[dict], annotations: dict[str, list[dict]]) -> list[Record]:
    out = []
    for r in rows:
        if r["dataset"] not in TWEETS or r["of_id"] != "-1":
            continue
        notes = annotations[r["id"]]
        votes = Counter(level(a) for a in notes)
        hate = next((lv for lv, n in votes.items() if n * 2 > len(notes)), "unknown")
        voters = [a for a in notes if sexist(a)]
        out.append(Record(
            source="cmsb", source_id=r["id"], platform="twitter", text=r["text"], text_form="preprocessed", hate=hate,
            label_raw=f"{r['dataset']} sexist={r['sexist']} "
                      + "|".join(f"{a['content']}/{a['phrasing']}" for a in notes),
            target_category=("gender",) if hate == "hateful" else (), target_group=(), n_annotators=len(notes),
            soft_label={lv: votes[lv] / len(notes) for lv in SOFT_LEVELS}, licence=LICENCE,
            explicitness=("explicit" if sum(a["phrasing"] == "1" for a in voters) * 2 > len(voters) else "implicit")
            if hate == "hateful" else "unknown"))
    return out


def check(rows: list[dict], annotations: dict[str, list[dict]], out: list[Record]) -> None:
    """Fails unless every kept tweet has five annotations and the authors' sexist label is the majority of annotators
    marking sexist content or sexist phrasing; then prints the totals."""
    kept = [r for r in rows if r["dataset"] in TWEETS and r["of_id"] == "-1"]
    if any(len(annotations[r["id"]]) != 5 for r in kept):
        raise ValueError("a kept tweet does not have five annotations")
    if any((r["sexist"] == "True") != (sum(map(sexist, annotations[r["id"]])) >= 3) for r in kept):
        raise ValueError("an authors' sexist label is not the majority of its annotators")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    rows, annotations = fetch()
    out = records(rows, annotations)
    check(rows, annotations, out)
    return out


if __name__ == "__main__":
    build()
