# builder/sources/hasoc.py
"""HASOC 2019 (Mandl et al., 2019), English: 7,005 posts from Twitter and partly Facebook (each post's platform is not
given), labelled hate or offensive or neither, then hate, offensive or profane, from the organisers' site, which states
no licence. Hate is hateful; offensive and profane are offensive; the rest is neither. No targets. The training and test
files reuse ids for different posts, so each id carries its split. The overview paper's training table swaps the
offensive and profane counts against the file (667 and 451); both map to offensive. Some training posts hold line breaks
and apostrophes as escapes, which are decoded. The 2020 and 2021 English sets are password-protected and not yet
included.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.hasoc"""
import csv
import io
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from builder.schema import Record

URL = "https://hasocfire.github.io/hasoc/2019/files/english_dataset.zip"
CACHE = Path("data/sources/hasoc/english_dataset.zip")
FILES = {"train": "english_dataset/english_dataset.tsv", "test": "english_dataset/hasoc2019_en_test-2919.tsv"}
LICENCE = "unspecified"
CITATION = ("Mandl, T., Modha, S., Majumder, P., Patel, D., Dave, M., Mandlia, C. and Patel, A. (2019). Overview of "
            "the HASOC track at FIRE 2019: Hate Speech and Offensive Content Identification in Indo-European "
            "Languages. FIRE 2019. https://doi.org/10.1145/3368567.3368584")
LEVELS = {"HATE": "hateful", "OFFN": "offensive", "PRFN": "offensive", "NONE": "neither"}
TABLES = {  # the overview paper's Tables 2 and 3: NOT, HOF, HATE, and offensive plus profane together
    "train": (3591, 2261, 1143, 1118), "test": (865, 288, 124, 164),
}
ESCAPED = re.compile(r"\\([n'])")  # the files' only escapes; a shrug's backslash is not one


def fetch(cache: Path = CACHE) -> list[dict]:
    """The training and test posts, with their split, from the organisers' zip, downloaded once."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    rows = []
    with zipfile.ZipFile(cache) as z:
        for split, name in FILES.items():
            with io.TextIOWrapper(z.open(name), encoding="utf-8", newline="") as f:
                rows += [{**r, "split": split} for r in csv.DictReader(f, delimiter="\t")]
    return rows


def records(rows: list[dict]) -> list[Record]:
    return [Record(
        source="hasoc", source_id=f"{r['split']}_{r['text_id']}", platform="unknown",
        text=ESCAPED.sub(lambda m: "\n" if m[1] == "n" else "'", r["text"]), text_form="raw",
        hate=LEVELS[r["task_2"]], label_raw=f"{r['task_1']}/{r['task_2']}/{r['task_3']}", target_category=(),
        target_group=(), n_annotators=None, soft_label=None, licence=LICENCE) for r in rows]


def check(rows: list[dict], out: list[Record]) -> None:
    """Fails unless each split reproduces the overview paper's counts, the two tasks agree (hate or offensive exactly
    where a fine label is given) and no escape is left in a post; then prints the totals."""
    for split, published in TABLES.items():
        part = [r for r in rows if r["split"] == split]
        task1, task2 = Counter(r["task_1"] for r in part), Counter(r["task_2"] for r in part)
        if (task1["NOT"], task1["HOF"], task2["HATE"], task2["OFFN"] + task2["PRFN"]) != published:
            raise ValueError(f"{split}: {dict(task1)} {dict(task2)}; the paper gives {published}")
    if any((r["task_1"] == "HOF") != (r["task_2"] != "NONE") for r in rows):
        raise ValueError("a post's two tasks disagree")
    if escaped := sum(bool(ESCAPED.search(r.text)) for r in out):
        raise ValueError(f"{escaped} posts still hold escapes")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    rows = fetch()
    out = records(rows)
    check(rows, out)
    return out


if __name__ == "__main__":
    build()
