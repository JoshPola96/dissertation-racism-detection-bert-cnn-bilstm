# builder/sources/hatexplain.py
"""HateXplain (Mathew et al., 2021): 20,148 Twitter and Gab posts, each labelled by three annotators as hate speech,
offensive or normal, with the groups each annotator saw targeted. Fetched from the authors' repository (CC BY 4.0).
The text is the authors' tokens, lower-cased, with placeholders such as <user> and <number>.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.hatexplain"""
import json
import urllib.request
from collections import Counter
from pathlib import Path

from builder.schema import Record

URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
CACHE = Path("data/sources/hatexplain/dataset.json")
LICENCE = "CC-BY-4.0"
CITATION = ("Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P. and Mukherjee, A. (2021). HateXplain: A "
            "Benchmark Dataset for Explainable Hate Speech Detection. AAAI 2021. "
            "https://doi.org/10.1609/aaai.v35i17.17745")
LEVELS = {"hatespeech": "hateful", "offensive": "offensive", "normal": "neither"}
GROUPS = {  # HateXplain's target names: (our target categories, our target group); "None" means no target
    "African": (("race_ethnicity",), "black"),
    "Arab": (("race_ethnicity",), "arab"),
    "Asian": (("race_ethnicity",), "asian"),
    "Caucasian": (("race_ethnicity",), "white"),
    "Hispanic": (("race_ethnicity",), "latinx"),
    "Indigenous": (("race_ethnicity",), "indigenous"),
    "Indian": (("race_ethnicity", "national_origin"), "indian"),
    "Jewish": (("race_ethnicity", "religion"), "jewish"),  # an ethnic and a religious group: both categories
    "Islam": (("religion",), "muslim"),
    "Christian": (("religion",), "christian"),
    "Hindu": (("religion",), "hindu"),
    "Buddhism": (("religion",), "buddhist"),
    "Nonreligious": (("religion",), "nonreligious"),
    "Refugee": (("national_origin",), "refugees"),
    "Women": (("gender",), "women"),
    "Men": (("gender",), "men"),
    "Homosexual": (("sexual_orientation",), "gay"),
    "Heterosexual": (("sexual_orientation",), "heterosexual"),
    "Bisexual": (("sexual_orientation",), "bisexual"),
    "Asexual": (("sexual_orientation",), "asexual"),
    "Disability": (("disability",), "disabled"),
    "Economic": (("other",), "economic"),
    "Minority": (("other",), "minority"),
    "Other": (("other",), None),
}
PUBLISHED = {  # Table 4 of the paper: posts per platform and majority-vote class; its "Undecided" is our "unknown"
    "twitter": {"hateful": 708, "offensive": 2328, "neither": 5770, "unknown": 249},
    "gab": {"hateful": 5227, "offensive": 3152, "neither": 2044, "unknown": 670},
}


def fetch(cache: Path = CACHE) -> dict:
    """The authors' dataset.json, downloaded once and read from the cache after that."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    return json.loads(cache.read_text(encoding="utf-8"))


def aggregate(annotators: list[dict]) -> tuple[str, tuple[str, ...], dict[str, float]]:
    """One post's three annotations combined into our hate level, the HateXplain target names kept for the post, and
    the soft label (the share of annotators who chose each of our levels)."""
    votes = Counter(LEVELS[a["label"]] for a in annotators)
    top, n = votes.most_common(1)[0]
    hate = top if n * 2 > len(annotators) else "unknown"
    target_votes = Counter(t for a in annotators for t in a["target"] if t != "None")
    targets = tuple(sorted(t for t, v in target_votes.items() if v * 2 > len(annotators)))
    soft = {level: votes[level] / len(annotators) for level in LEVELS.values()}
    return hate, targets, soft


def records(posts: dict) -> list[Record]:
    out = []
    for post_id, post in posts.items():
        hate, targets, soft = aggregate(post["annotators"])
        platform = "gab" if post_id.endswith("_gab") else "twitter"  # Table 4 counts the 28 other ids as Twitter
        out.append(Record(
            source="hatexplain", source_id=post_id, platform=platform,
            text=" ".join(post["post_tokens"]), text_form="preprocessed", hate=hate,
            label_raw="|".join(a["label"] for a in post["annotators"]),
            target_category=tuple(sorted({c for t in targets for c in GROUPS[t][0]})),
            target_group=tuple(sorted({GROUPS[t][1] for t in targets} - {None})),
            n_annotators=len(post["annotators"]), soft_label=soft, licence=LICENCE))
    return out


def check(posts: dict, out: list[Record]) -> None:
    """Fails unless every target name has a mapping and the build reproduces the paper's counts for each platform and
    level, then prints the totals."""
    if missing := {t for p in posts.values() for a in p["annotators"] for t in a["target"]} - GROUPS.keys() - {"None"}:
        raise ValueError(f"target names with no mapping: {missing}")
    for platform, published in PUBLISHED.items():
        if (built := dict(Counter(r.hate for r in out if r.platform == platform))) != published:
            raise ValueError(f"{platform}: built {built}, published {published}")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    posts = fetch()
    out = records(posts)
    check(posts, out)
    return out


if __name__ == "__main__":
    build()
