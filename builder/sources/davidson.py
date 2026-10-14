# builder/sources/davidson.py
"""Davidson et al. (2017): 24,783 tweets, each labelled by three or more annotators as hate speech, offensive language
or neither, with every vote released, from the authors' GitHub (MIT). The level is the strict majority of votes
(unknown for the 60 tweets with only a plurality) and the soft label is the votes' shares. No targets. The release is
not the paper's sample: the paper describes 24,802 tweets, 1.3% of them unanimously hate, where the file holds 24,783
and 1.06%.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.davidson"""
import csv
import urllib.request
from collections import Counter
from pathlib import Path

from builder.schema import SOFT_LEVELS, Record

URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
CACHE = Path("data/sources/davidson/labeled_data.csv")
LICENCE = "MIT"
CITATION = ("Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017). Automated Hate Speech Detection and the "
            "Problem of Offensive Language. ICWSM 2017. https://doi.org/10.1609/icwsm.v11i1.14955")
VOTES = {"hateful": "hate_speech", "offensive": "offensive_language", "neither": "neither"}
CLASSES = {"0": "hateful", "1": "offensive", "2": "neither"}


def fetch(cache: Path = CACHE) -> list[dict]:
    """The authors' labelled tweets, downloaded once."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    with cache.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def records(rows: list[dict]) -> list[Record]:
    out = []
    for r in rows:
        votes, total = {lv: int(r[col]) for lv, col in VOTES.items()}, int(r["count"])
        out.append(Record(
            source="davidson", source_id=r[""], platform="twitter", text=r["tweet"], text_form="raw",
            hate=next((lv for lv, n in votes.items() if n * 2 > total), "unknown"),
            label_raw=f"hate_speech={r['hate_speech']} offensive_language={r['offensive_language']} "
                      f"neither={r['neither']} class={r['class']}",
            target_category=(), target_group=(), n_annotators=total,
            soft_label={lv: votes[lv] / total for lv in SOFT_LEVELS}, licence=LICENCE))
    return out


def check(rows: list[dict], out: list[Record]) -> None:
    """Fails unless the file holds the 24,783 tweets of the authors' own dataset card, every tweet's votes add up to its
    annotator count, and the authors' class is the plurality of the votes; then prints the totals."""
    if len(rows) != 24783:
        raise ValueError(f"{len(rows)} tweets; the authors' card gives 24,783")
    if any(sum(int(r[col]) for col in VOTES.values()) != int(r["count"]) for r in rows):
        raise ValueError("a tweet's votes do not add up to its annotator count")
    if any(CLASSES[r["class"]] != max(VOTES, key=lambda lv: int(r[VOTES[lv]])) for r in rows):
        raise ValueError("an authors' class is not the plurality of its votes")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    rows = fetch()
    out = records(rows)
    check(rows, out)
    return out


if __name__ == "__main__":
    build()
