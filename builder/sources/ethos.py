# builder/sources/ethos.py
"""ETHOS (Mollas et al., 2022): 998 YouTube and Reddit comments, each with the share of annotators who judged it hate
speech, and target categories for the 433 the authors count as hate, from the authors' GitHub (GPL-3.0). Hateful when
that share is above one half, unknown at exactly one half (the authors count it as hate), and not hateful below it,
since ETHOS asks only about hate and so does not separate offensive from neutral. Targets are kept when more than half
the annotators named them. The platform of each comment is not given, and the annotators' count varies and is not
released.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.ethos"""
import csv
import urllib.request
from collections import Counter
from pathlib import Path

from builder.schema import Record

URL = "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/"
CACHE = Path("data/sources/ethos")
FILES = ("Ethos_Dataset_Binary.csv", "Ethos_Dataset_Multi_Label.csv")
LICENCE = "GPL-3.0"
CITATION = ("Mollas, I., Chrysopoulou, Z., Karlos, S. and Tsoumakas, G. (2022). ETHOS: a multi-label hate speech "
            "detection dataset. Complex & Intelligent Systems. https://doi.org/10.1007/s40747-021-00608-2")
CATEGORIES = {"race": "race_ethnicity", "national_origin": "national_origin", "religion": "religion",
              "gender": "gender", "sexual_orientation": "sexual_orientation", "disability": "disability"}


def fetch(cache: Path = CACHE) -> tuple[list[dict], list[dict]]:
    """The binary and the multi-label files, downloaded once."""
    tables = []
    for name in FILES:
        if not (cache / name).exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL + name, cache / name)
        with (cache / name).open(encoding="utf-8", newline="") as f:
            tables.append(list(csv.DictReader(f, delimiter=";")))
    return tables[0], tables[1]


def key(text: str) -> str:
    """The two files space the same comment differently (non-breaking spaces in one), so they join on this."""
    return " ".join(text.split())


def records(binary: list[dict], multi: list[dict]) -> list[Record]:
    labels = {key(r["comment"]): r for r in multi}
    out = []
    for i, r in enumerate(binary):
        share = float(r["isHate"])
        found = labels.get(key(r["comment"]), {})
        out.append(Record(
            source="ethos", source_id=str(i), platform="unknown", text=r["comment"], text_form="raw",
            hate="hateful" if share > 0.5 else "unknown" if share == 0.5 else "not_hateful",
            label_raw=f"isHate={r['isHate']}",
            target_category=tuple(sorted(c for col, c in CATEGORIES.items() if float(found.get(col, 0)) > 0.5)),
            target_group=(), n_annotators=None, soft_label=None, licence=LICENCE))
    return out


def check(binary: list[dict], multi: list[dict], out: list[Record]) -> None:
    """Fails unless the files hold the authors' 998 comments, 433 of them hate at the authors' cut of one half, and
    every multi-label comment joins to one binary comment; then prints the totals."""
    if (counts := (len(binary), sum(float(r["isHate"]) >= 0.5 for r in binary), len(multi))) != (998, 433, 433):
        raise ValueError(f"comments, hate, multi-label: {counts}; the authors give (998, 433, 433)")
    if unjoined := {key(r["comment"]) for r in multi} - {key(r["comment"]) for r in binary}:
        raise ValueError(f"{len(unjoined)} multi-label comments match no binary comment")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    binary, multi = fetch()
    out = records(binary, multi)
    check(binary, multi, out)
    return out


if __name__ == "__main__":
    build()
