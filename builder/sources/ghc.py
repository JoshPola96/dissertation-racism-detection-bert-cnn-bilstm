# builder/sources/ghc.py
"""Gab Hate Corpus (Kennedy et al., 2022): 27,665 Gab posts with 86,529 annotations, one row per annotator, from the
authors' OSF project (CC BY 4.0). Each annotator marks hate (an assault on human dignity or a call for violence) and,
independently, vulgar or offensive language; targets only when they marked hate. So targets exist only on posts a
majority called hateful, and the corpus holds no non-hateful posts that mention a group. Explicitness is left unknown:
the authors report too little agreement on it to use. GHC names target categories only, never finer groups.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.ghc"""
import csv
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from builder.schema import SOFT_LEVELS, Record

URL = "https://osf.io/download/wuecz/"
CACHE = Path("data/sources/ghc/GabHateCorpus_annotations.tsv")
LICENCE = "CC-BY-4.0"
CITATION = ("Kennedy, B. et al. (2022). Introducing the Gab Hate Corpus: Defining and Applying Hate-Based Rhetoric "
            "to Social Media Posts at Scale. Language Resources and Evaluation. "
            "https://doi.org/10.1007/s10579-021-09569-x")
TARGETS = {"RAE": "race_ethnicity", "NAT": "national_origin", "REL": "religion", "GEN": "gender",
           "SXO": "sexual_orientation", "IDL": "political", "POL": "political", "MPH": "disability"}
PUBLISHED = {  # the authors' train and test files: posts per (HD, CV, VO), majority vote with ties as positive
    (0, 0, 0): 24169, (0, 0, 1): 927, (0, 1, 0): 64, (0, 1, 1): 38,
    (1, 0, 0): 1538, (1, 0, 1): 757, (1, 1, 0): 27, (1, 1, 1): 26,
}


def fetch(cache: Path = CACHE) -> dict[str, list[dict]]:
    """The annotations file, downloaded once, grouped into one list of annotator rows per post ID."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    posts = defaultdict(list)
    with cache.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            posts[row["ID"]].append(row)
    return posts


def yes(value: str) -> bool:
    """The file writes "yes" as '1' in the label columns but '1.0' in the target columns, and leaves targets blank
    when an annotator did not mark hate."""
    return value in ("1", "1.0")


def margin(rows: list[dict], column: str) -> int:
    """How many more annotators answered yes than no: positive is a majority, zero a tie."""
    return sum(1 if yes(r[column]) else -1 for r in rows)


def aggregate(rows: list[dict]) -> tuple[str, tuple[str, ...], dict[str, float]]:
    """One post's annotator rows combined into our hate level, the target categories named by a majority of the
    annotators who marked hate, and the soft label. Hate and vulgar or offensive language were separate yes/no
    questions, so each gets its own majority: hateful on a hate majority, offensive on a no-hate and a yes-vulgarity
    majority, neither on two no majorities, unknown on a tie."""
    hate, vo = margin(rows, "Hate"), margin(rows, "VO")
    level = "hateful" if hate > 0 else "unknown" if hate == 0 or vo == 0 else "offensive" if vo > 0 else "neither"
    asked = [r for r in rows if yes(r["Hate"])] if level == "hateful" else []
    targets = tuple(sorted({TARGETS[c] for c in TARGETS if margin(asked, c) > 0}))
    votes = Counter("hateful" if yes(r["Hate"]) else "offensive" if yes(r["VO"]) else "neither" for r in rows)
    soft = {lv: votes[lv] / len(rows) for lv in SOFT_LEVELS}
    return level, targets, soft


def records(posts: dict[str, list[dict]]) -> list[Record]:
    out = []
    for post_id, rows in posts.items():
        hate, targets, soft = aggregate(rows)
        out.append(Record(
            source="ghc", source_id=post_id, platform="gab", text=rows[0]["Text"], text_form="raw", hate=hate,
            label_raw="|".join(f"Hate={r['Hate']} VO={r['VO']}" for r in rows),
            target_category=targets, target_group=(), n_annotators=len(rows), soft_label=soft, licence=LICENCE))
    return out


def check(posts: dict[str, list[dict]], out: list[Record]) -> None:
    """Fails unless every value parsed is one the file is known to use, the posts number the published 27,665, and the
    grouped annotations reproduce the authors' released per-post labels; then prints the totals."""
    if unexpected := {r[c] for rows in posts.values() for r in rows for c in ("Hate", "HD", "CV", "VO", *TARGETS)} - {
            "0", "1", "", "0.0", "1.0"}:
        raise ValueError(f"values the parser does not know: {unexpected}")
    if len(posts) != 27665:
        raise ValueError(f"{len(posts)} posts, published 27,665")
    released = Counter(tuple(int(margin(rows, c) >= 0) for c in ("HD", "CV", "VO"))
                       for rows in posts.values() if len(rows) >= 3)  # the authors left two-annotator posts out
    if dict(released) != PUBLISHED:
        raise ValueError(f"grouped labels {dict(released)} differ from the released {PUBLISHED}")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    posts = fetch()
    out = records(posts)
    check(posts, out)
    return out


if __name__ == "__main__":
    build()
