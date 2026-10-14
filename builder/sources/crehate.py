# builder/sources/crehate.py
"""CREHate (Lee et al., 2024): 1,580 posts, 980 re-annotated from Social Bias Frames and 600 collected from Reddit and
YouTube in Australia, the United Kingdom, Singapore and South Africa, each labelled hate or not by annotators from those
four countries and the United States, from the Hugging Face Hub (CC BY-SA 4.0). Each country's label is the majority of
its annotators, "not sure" abstaining; each country then has one vote, so a post is hateful when three or more countries
call it hate and not hateful otherwise (the dataset asks only about hate). The five country labels are kept as the raw
label. No targets. The re-annotated posts carry no platform here; their Social Bias Frames copies do.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.crehate"""
import csv
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from builder.schema import Record

URL = "https://huggingface.co/datasets/nayeon212/CREHate/resolve/main/data/"
CACHE = Path("data/sources/crehate")
FILES = ("CREHate_final_annotations.csv", "CREHate_raw_annotations.csv")
LICENCE = "CC-BY-SA-4.0"
CITATION = ("Lee, N., Jung, C., Myung, J., Jin, J., Camacho-Collados, J., Kim, J. and Oh, A. (2024). Exploring "
            "Cross-Cultural Differences in English Hate Speech Annotations: From Dataset Construction to Analysis. "
            "NAACL 2024. https://doi.org/10.18653/v1/2024.naacl-long.236")
COUNTRIES = ("United_States", "Australia", "United_Kingdom", "South_Africa", "Singapore")
CARD = {"SBIC": 980, "Reddit": 311, "YouTube": 289}  # the dataset card's posts by division and source


def fetch(cache: Path = CACHE) -> tuple[list[dict], list[dict]]:
    """The posts with their country labels, and every annotation, downloaded once."""
    csv.field_size_limit(sys.maxsize)
    tables = []
    for name in FILES:
        if not (cache / name).exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL + name, cache / name)
        with (cache / name).open(encoding="utf-8", newline="") as f:
            tables.append(list(csv.DictReader(f)))
    return tables[0], tables[1]


def platform(post_id: str) -> str:
    return "unknown" if post_id.startswith("SBIC") else post_id.split("-")[1].lower()


def records(posts: list[dict], annotations: list[dict]) -> list[Record]:
    counted = Counter(a["ID"] for a in annotations)
    return [Record(
        source="crehate", source_id=p["ID"], platform=platform(p["ID"]), text=p["Text"], text_form="raw",
        hate="hateful" if sum(p[f"{c}_Hate"] == "1.0" for c in COUNTRIES) >= 3 else "not_hateful",
        label_raw=" ".join(f"{c}={p[f'{c}_Hate'][0]}" for c in COUNTRIES), target_category=(), target_group=(),
        n_annotators=counted[p["ID"]], soft_label=None, licence=LICENCE) for p in posts]


def check(posts: list[dict], annotations: list[dict], out: list[Record]) -> None:
    """Fails unless the posts reproduce the card's counts by division and source and the paper's 56.2% all-country
    consensus (888 posts), and every country label is the majority of that country's annotators with "not sure" (2)
    abstaining; then prints the totals."""
    sources = Counter("SBIC" if p["ID"].startswith("SBIC") else p["ID"].split("-")[1] for p in posts)
    if dict(sources) != CARD or sum(len({p[f"{c}_Hate"] for c in COUNTRIES}) == 1 for p in posts) != 888:
        raise ValueError(f"posts by source {dict(sources)}; the card gives {CARD} and 888 posts in consensus")
    votes = defaultdict(list)
    for a in annotations:
        if a["annotation"] != "2":
            votes[a["ID"], a["nationality"].replace(" ", "_")].append(a["annotation"] == "1")
    if any((sum(votes[p["ID"], c]) * 2 > len(votes[p["ID"], c])) != (p[f"{c}_Hate"] == "1.0")
           for p in posts for c in COUNTRIES):
        raise ValueError("a country label is not the majority of that country's annotators")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    posts, annotations = fetch()
    out = records(posts, annotations)
    check(posts, annotations, out)
    return out


if __name__ == "__main__":
    build()
