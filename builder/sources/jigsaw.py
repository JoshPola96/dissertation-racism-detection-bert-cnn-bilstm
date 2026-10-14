# builder/sources/jigsaw.py
"""Jigsaw Unintended Bias (Civil Comments, 2019): of 1,999,516 news-site comments, the 448,000 whose identity mentions
were annotated, from the Kaggle competition, downloaded by hand once its rules are accepted (CC0, text included). Each
score is the share of annotators who said yes. Hateful when more than half call the comment an identity attack, else
offensive when more than half call it toxic, neither when fewer than half say either, and unknown at exactly half.
Identities more than half the annotators saw mentioned are kept as targets, at every level. The other comments were
never annotated for identities, so a blank there means "not asked", and they are left out.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.jigsaw"""
import csv
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

from builder.schema import Record

COMPETITION = "jigsaw-unintended-bias-in-toxicity-classification"
CACHE = Path("data/sources/jigsaw/all_data.csv.zip")
LICENCE = "CC0-1.0"
CITATION = ("Borkan, D., Dixon, L., Sorensen, J., Thain, N. and Vasserman, L. (2019). Nuanced Metrics for Measuring "
            "Unintended Bias with Real Data for Text Classification. WWW '19 Companion. "
            "https://doi.org/10.1145/3308560.3317593")
IDENTITIES = {  # identity column: (our target categories, our target group)
    "male": (("gender",), "men"), "female": (("gender",), "women"), "transgender": (("gender",), "transgender"),
    "other_gender": (("gender",), None), "heterosexual": (("sexual_orientation",), "heterosexual"),
    "homosexual_gay_or_lesbian": (("sexual_orientation",), "gay"), "bisexual": (("sexual_orientation",), "bisexual"),
    "other_sexual_orientation": (("sexual_orientation",), None), "christian": (("religion",), "christian"),
    "jewish": (("race_ethnicity", "religion"), "jewish"), "muslim": (("religion",), "muslim"),
    "hindu": (("religion",), "hindu"), "buddhist": (("religion",), "buddhist"),
    "atheist": (("religion",), "nonreligious"),
    "other_religion": (("religion",), None), "black": (("race_ethnicity",), "black"),
    "white": (("race_ethnicity",), "white"), "asian": (("race_ethnicity",), "asian"),
    "latino": (("race_ethnicity",), "latinx"), "other_race_or_ethnicity": (("race_ethnicity",), None),
    "physical_disability": (("disability",), "physical_disability"),
    "intellectual_or_learning_disability": (("disability",), "cognitive_disability"),
    "psychiatric_or_mental_illness": (("disability",), "mental_illness"), "other_disability": (("disability",), None),
}


def fetch(cache: Path = CACHE) -> tuple[list[dict], int]:
    """The identity-annotated comments from the competition's all_data.csv, and how many comments the file holds."""
    if not cache.exists():
        raise FileNotFoundError(f"{cache}: accept the rules of the Kaggle competition {COMPETITION}, then run "
                                f"kaggle competitions download {COMPETITION} -f all_data.csv -p {cache.parent}")
    csv.field_size_limit(sys.maxsize)
    rows, total = [], 0
    with zipfile.ZipFile(cache) as z, io.TextIOWrapper(z.open("all_data.csv"), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if int(row["identity_annotator_count"]) > 0:
                rows.append(row)
    return rows, total


def level(toxicity: float, attack: float) -> str:
    if attack > 0.5:
        return "hateful"
    if attack == 0.5 or toxicity == 0.5:
        return "unknown"
    return "offensive" if toxicity > 0.5 else "neither"


def records(rows: list[dict]) -> list[Record]:
    out = []
    for r in rows:
        named = [IDENTITIES[c] for c in IDENTITIES if float(r[c]) > 0.5]
        out.append(Record(
            source="jigsaw", source_id=r["id"], platform="civil_comments", text=r["comment_text"], text_form="raw",
            hate=level(float(r["toxicity"]), float(r["identity_attack"])),
            label_raw=f"toxicity={r['toxicity']} identity_attack={r['identity_attack']}",
            target_category=tuple(sorted({c for cats, _ in named for c in cats})),
            target_group=tuple(sorted({g for _, g in named if g})),
            n_annotators=int(r["toxicity_annotator_count"]), soft_label=None, licence=LICENCE))
    return out


def check(rows: list[dict], total: int, out: list[Record]) -> None:
    """Fails unless the file holds the competition's 1,999,516 comments, 448,000 of them identity-annotated (the
    WILDS CivilComments count), with every identity score present; then prints the totals."""
    if (counts := (total, len(rows))) != (1999516, 448000):
        raise ValueError(f"comments, identity-annotated: {counts}; expected (1999516, 448000)")
    if any(r[c] == "" for r in rows for c in IDENTITIES):
        raise ValueError("an identity-annotated comment has a blank identity score")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    rows, total = fetch()
    out = records(rows)
    check(rows, total, out)
    return out


if __name__ == "__main__":
    build()
