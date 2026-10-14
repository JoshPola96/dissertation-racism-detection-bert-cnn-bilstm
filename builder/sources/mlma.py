# builder/sources/mlma.py
"""MLMA (Ousidhoum et al., 2019), English part: 5,647 tweets from the authors' GitHub (MIT). Each tweet carries the
hostility types at least two annotators agreed on. Hateful when one is "hateful", offensive for any other hostility
type, neither when the only type is "normal", and unknown when "normal" sits beside a hostility type, since the
annotators then disagreed whether the tweet is hostile at all. The authors' directness label is their explicitness
(direct is explicit), given for hostile tweets. Targets were asked for every tweet. The authors replaced mentions and
links with @user and @URL, and stored each tweet JSON-escaped with its double quotes deleted; the escapes are decoded
and the quotes put back. The per-annotator labels are lost, so no soft label.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.mlma"""
import csv
import io
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from builder.schema import Record

URL = "https://github.com/HKUST-KnowComp/MLMA_hate_speech/raw/master/hate_speech_mlma.zip"
CACHE = Path("data/sources/mlma/hate_speech_mlma.zip")
MEMBER = "hate_speech_mlma/en_dataset_with_stop_words.csv"
LICENCE = "MIT"
CITATION = ("Ousidhoum, N., Lin, Z., Zhang, H., Song, Y. and Yeung, D.-Y. (2019). Multilingual and Multi-Aspect Hate "
            "Speech Analysis. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1474")
ATTRIBUTES = {"origin": "national_origin", "gender": "gender", "sexual_orientation": "sexual_orientation",
              "religion": "religion", "disability": "disability", "other": "other"}
GROUPS = {  # the target group, where it names one: (our target categories, our target group)
    "african_descent": (("race_ethnicity",), "black"), "hispanics": (("race_ethnicity",), "latinx"),
    "asians": (("race_ethnicity",), "asian"), "arabs": (("race_ethnicity",), "arab"),
    "immigrants": (("national_origin",), "immigrants"), "refugees": (("national_origin",), "refugees"),
    "indian/hindu": (("national_origin", "religion"), "indian"), "muslims": (("religion",), "muslim"),
    "jews": (("race_ethnicity", "religion"), "jewish"), "christian": (("religion",), "christian"),
    "women": (("gender",), "women"), "gay": (("sexual_orientation",), "gay"),
    "special_needs": (("disability",), "disabled"), "left_wing_people": (("political",), "liberals"),
}
TABLE_2 = {  # the paper's English counts: tweets per hostility type, and per target attribute
    "hostility": {"abusive": 671, "hateful": 1278, "offensive": 4020, "disrespectful": 782, "fearful": 562,
                  "normal": 1359},
    "target": {"origin": 2448, "gender": 638, "sexual_orientation": 514, "religion": 68, "disability": 1089,
               "other": 890},
}
ESCAPE = re.compile(r"(?<!\S)\\(?=n[a-z][^\\]*\\(?![a-zA-Z]))|\\(\\(?![nu])|u[0-9a-fA-F]{4}|n|)")


def fetch(cache: Path = CACHE) -> list[dict]:
    """The English file from the authors' zip, downloaded once."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    with zipfile.ZipFile(cache) as z, io.TextIOWrapper(z.open(MEMBER), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unescape(text: str) -> str:
    """The tweet as posted. Deleting the quotes left each escaped quote as a lone backslash, so a backslash is a quote
    unless it escapes a backslash (not one before a line break or character code: that pair is a closing quote and the
    escape), a line break or a character code; a line-break escape that opens a quoted word (\\nonce\\) is the quote."""
    def decode(match: re.Match) -> str:
        escape = match.group(1)
        return '"' if not escape else "\n" if escape == "n" else "\\" if escape == "\\" else chr(int(escape[1:], 16))
    return ESCAPE.sub(decode, text).encode("utf-16", "surrogatepass").decode("utf-16")


def level(types: set[str]) -> str:
    if types == {"normal"}:
        return "neither"
    return "unknown" if "normal" in types else "hateful" if "hateful" in types else "offensive"


def records(rows: list[dict]) -> list[Record]:
    out = []
    for r in rows:
        types = set(r["sentiment"].split("_"))
        categories, group = GROUPS.get(r["group"], ((ATTRIBUTES[r["target"]],), None))
        out.append(Record(
            source="mlma", source_id=r["HITId"], platform="twitter", text=unescape(r["tweet"]), text_form="preprocessed",
            hate=level(types), label_raw=f"{r['sentiment']} {r['directness']}",
            target_category=tuple(sorted(categories)), target_group=(group,) if group else (), n_annotators=None,
            soft_label=None, licence=LICENCE, explicitness="unknown" if types == {"normal"}
            else "explicit" if r["directness"] == "direct" else "implicit"))
    return out


def check(rows: list[dict], out: list[Record]) -> None:
    """Fails unless the tweets reproduce the paper's English counts, every target attribute and group is mapped and no
    line-break or character-code escape is left in a tweet; then prints the totals."""
    built = {"hostility": dict(Counter(t for r in rows for t in r["sentiment"].split("_"))),
             "target": dict(Counter(r["target"] for r in rows))}
    if len(rows) != 5647 or built != TABLE_2:
        raise ValueError(f"{len(rows)} tweets, counts {built}; the paper gives 5,647 tweets and {TABLE_2}")
    if missing := {r["group"] for r in rows} - GROUPS.keys() - {"individual", "other"}:
        raise ValueError(f"target groups with no mapping: {missing}")
    if escaped := sum(bool(re.search(r"\\(n|u[0-9a-fA-F]{4})", r.text)) for r in out):
        raise ValueError(f"{escaped} tweets still hold escapes")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    rows = fetch()
    out = records(rows)
    check(rows, out)
    return out


if __name__ == "__main__":
    build()
