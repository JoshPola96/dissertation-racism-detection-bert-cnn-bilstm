# builder/sources/foxnews.py
"""Fox News comments (Gao and Huang, 2017): 1,528 comments from 10 discussion threads on Fox News articles, labelled
hateful or not, from the authors' GitHub (MIT). Hateful is hateful; the rest is not hateful, since the dataset does
not separate offensive from neutral. No targets. Each line names its commenter, pseudonymised into an author id.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.foxnews"""
import re
import urllib.request
from collections import Counter
from pathlib import Path

from builder.privacy import pseudonym
from builder.schema import Record

URL = "https://raw.githubusercontent.com/sjtuprog/fox-news-comments/master/annotated-threads/all-comments.txt"
CACHE = Path("data/sources/foxnews/all-comments.txt")
LICENCE = "MIT"
CITATION = ("Gao, L. and Huang, R. (2017). Detecting Online Hate Speech Using Context Aware Models. RANLP 2017. "
            "https://aclanthology.org/R17-1036")
LINE = re.compile(r"([01]):(\S+) ?(.*)", re.DOTALL)  # label, then the commenter's name, then the comment


def fetch(cache: Path = CACHE) -> list[str]:
    """The comments, one per line, downloaded once."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    return [line for line in cache.read_text(encoding="utf-8").split("\n") if line.strip()]


def records(lines: list[str]) -> list[Record]:
    out = []
    for i, line in enumerate(lines, 1):
        label, user, text = LINE.fullmatch(line).groups()
        out.append(Record(
            source="foxnews", source_id=str(i), platform="fox_news", text=text, text_form="raw",
            hate="hateful" if label == "1" else "not_hateful", label_raw=label, target_category=(), target_group=(),
            n_annotators=None, soft_label=None, licence=LICENCE,
            author_id=pseudonym("fox_news", user)))
    return out


def check(lines: list[str], out: list[Record]) -> None:
    """Fails unless every line parses and the labels reproduce the authors' 435 hateful and 1,093 non-hateful
    comments; then prints the totals."""
    if not all(LINE.fullmatch(line) for line in lines):
        raise ValueError("a line does not read as label:commenter comment")
    if (counts := Counter(r.hate for r in out)) != Counter(hateful=435, not_hateful=1093):
        raise ValueError(f"levels {dict(counts)}; the authors give 435 hateful and 1,093 not")
    print("hate levels:", dict(counts))


def build() -> list[Record]:
    lines = fetch()
    out = records(lines)
    check(lines, out)
    return out


if __name__ == "__main__":
    build()
