# builder/sources/intervene.py
"""Learning to Intervene (Qian et al., 2019): Gab and Reddit conversations whose posts are labelled hate speech or not,
from the authors' GitHub (CC BY-NC 4.0). Conversations overlap, so a post can appear in several: it is hateful when
every appearance is labelled hate, not hateful (the dataset does not separate offensive from neutral) when none is, and
unknown when they disagree (257 posts). The written intervention responses are not included. No targets. The Reddit
file holds 15 fewer comments than the paper reports, and 2 more labelled hate.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.intervene"""
import ast
import csv
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from builder.schema import Record

URL = "https://raw.githubusercontent.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/"
CACHE = Path("data/sources/intervene")
LICENCE = "CC-BY-NC-4.0"
CITATION = ("Qian, J., Bethke, A., Liu, Y., Belding, E. and Wang, W. Y. (2019). A Benchmark Dataset for Learning to "
            "Intervene in Online Hate Speech. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1482")
LINE = re.compile(r"(?m)^(\d+)\. \t*")  # each post opens a line with its number, indented by its depth in the thread
PAPER = {  # conversations, post appearances, hate appearances, conversations with hate; Reddit's file differs on two
    "gab": (11825, 33776, 14614, 11169), "reddit": (5020, None, None, 3847),
}


def fetch(cache: Path = CACHE) -> dict[str, list[dict]]:
    """Each platform's conversations, downloaded once, as lists of numbered posts with their ids and hate flags."""
    csv.field_size_limit(sys.maxsize)
    conversations = {}
    for platform in PAPER:
        path = cache / f"{platform}.csv"
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{URL}master/data/{platform}.csv", path)
        with path.open(encoding="utf-8", newline="") as f:
            conversations[platform] = [{
                "numbers": LINE.split(r["text"])[1::2], "ids": [x.strip() for x in LINE.split(r["id"])[2::2]],
                "texts": [x.rstrip() for x in LINE.split(r["text"])[2::2]],
                "hate": set(ast.literal_eval(r["hate_speech_idx"])) if r["hate_speech_idx"].startswith("[") else set(),
            } for r in csv.DictReader(f)]
    return conversations


def records(conversations: dict[str, list[dict]]) -> list[Record]:
    out = []
    for platform, rows in conversations.items():
        texts, flags = {}, defaultdict(list)
        for c in rows:
            for i, (post_id, text) in enumerate(zip(c["ids"], c["texts"], strict=True), 1):
                texts[post_id] = text
                flags[post_id].append(i in c["hate"])
        for post_id, seen in flags.items():
            out.append(Record(
                source="intervene", source_id=f"{platform}_{post_id}", platform=platform, text=texts[post_id],
                text_form="raw", hate="hateful" if all(seen) else "not_hateful" if not any(seen) else "unknown",
                label_raw="|".join("hate" if s else "non-hate" for s in seen), target_category=(), target_group=(),
                n_annotators=None, soft_label=None, licence=LICENCE))
    return out


def check(conversations: dict[str, list[dict]], out: list[Record]) -> None:
    """Fails unless every conversation numbers its posts 1, 2, 3 and has an id for each, a post keeps one text across
    conversations, and the counts reproduce the paper's (all four for Gab; conversations, and those with hate, for
    Reddit); then prints the totals."""
    for platform, rows in conversations.items():
        if any(c["numbers"] != [str(i) for i in range(1, len(c["ids"]) + 1)] or len(c["texts"]) != len(c["ids"])
               for c in rows):
            raise ValueError(f"{platform}: a conversation's posts are misnumbered or lack ids")
        seen = defaultdict(set)
        for c in rows:
            for post_id, text in zip(c["ids"], c["texts"], strict=True):
                seen[post_id].add(text)
        if any(len(t) > 1 for t in seen.values()):
            raise ValueError(f"{platform}: a post has more than one text")
        built = (len(rows), sum(len(c["ids"]) for c in rows), sum(len(c["hate"]) for c in rows),
                 sum(bool(c["hate"]) for c in rows))
        if any(p is not None and p != b for p, b in zip(PAPER[platform], built, strict=True)):
            raise ValueError(f"{platform}: counts {built}; the paper gives {PAPER[platform]}")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    conversations = fetch()
    out = records(conversations)
    check(conversations, out)
    return out


if __name__ == "__main__":
    build()
