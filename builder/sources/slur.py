# builder/sources/slur.py
"""Slur Corpus (Kurrek et al., 2020): 40,000 Reddit comments, each using one of three slurs (against Black, gay and
transgender people), labelled by two annotators for how the slur is used, from the authors' GitHub (MIT). Derogatory
use is hateful; reclaimed use and homonyms are neither; the "non-derogatory, non-appropriative" use (discussing the
word, counter speech, sarcasm), which keeps the slur's force, is not hateful; noise is unknown. Every use but a homonym
targets the slur's group. Authors are pseudonymised as for CAD, so one Reddit user has one id across both.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.slur"""
import json
import urllib.request
from collections import Counter
from pathlib import Path

from builder.privacy import pseudonym
from builder.schema import Record

URL = "https://raw.githubusercontent.com/networkdynamics/slur-corpus/main/kurrek.2020.slur-corpus.json"
CACHE = Path("data/sources/slur/kurrek.2020.slur-corpus.json")
LICENCE = "MIT"
CITATION = ("Kurrek, J., Saleem, H. M. and Ruths, D. (2020). Towards a Comprehensive Taxonomy and Large-Scale "
            "Annotated Corpus for Online Slur Usage. Fourth Workshop on Online Abuse and Harms. "
            "https://aclanthology.org/2020.alw-1.17")
LEVELS = {"DEG": "hateful", "NDG": "not_hateful", "APR": "neither", "HOM": "neither", "CMP": "unknown"}
SLURS = {"nigger": ("race_ethnicity", "black"), "faggot": ("sexual_orientation", "gay"),
         "tranny": ("gender", "transgender")}
README = {"DEG": 20531, "NDG": 16729, "HOM": 1998, "APR": 553, "CMP": 189}  # the authors' label counts


def fetch(cache: Path = CACHE) -> list[dict]:
    """The corpus, one JSON record per line, downloaded once. The CSV twin splits a few comments across rows."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    with cache.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def records(rows: list[dict]) -> list[Record]:
    out = []
    for r in rows:
        category, group = SLURS[r["slur"]] if r["gold_label"] != "HOM" else (None, None)
        out.append(Record(
            source="slur", source_id=r["id"], platform="reddit", text=r["body"], text_form="raw",
            hate=LEVELS[r["gold_label"]], label_raw=f"{r['gold_label']} disagreement={r['disagreement']}",
            target_category=(category,) if category else (), target_group=(group,) if group else (),
            n_annotators=2, soft_label=None, licence=LICENCE,
            author_id=pseudonym("reddit", r["author"])))
    return out


def check(rows: list[dict], out: list[Record]) -> None:
    """Fails unless the labels reproduce the authors' counts, every comment has its own id and one of the three slurs;
    then prints the totals."""
    if (built := dict(Counter(r["gold_label"] for r in rows))) != README:
        raise ValueError(f"labels {built}; the authors give {README}")
    if len({r["id"] for r in rows}) != len(rows) or any(r["slur"] not in SLURS for r in rows):
        raise ValueError("a repeated comment id or an unknown slur")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    rows = fetch()
    out = records(rows)
    check(rows, out)
    return out


if __name__ == "__main__":
    build()
