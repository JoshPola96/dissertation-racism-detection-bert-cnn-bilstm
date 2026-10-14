# builder/sources/mhs.py
"""Measuring Hate Speech (Kennedy et al., 2020; Sachdeva et al., 2022): 39,565 Reddit, Twitter and YouTube comments
with 135,556 annotations by 7,912 annotators, from the Hugging Face Hub (CC BY 4.0). The hate level comes from the
authors' hate speech score, a Rasch measure adjusted for each annotator's severity: above 0.5 is hateful, below -1 is
counter or supportive speech (neither), and the band between, which the authors call neutral or ambiguous, is unknown,
since the corpus does not separate offensive from neutral. Every annotator was asked for targets, so targets are kept
by majority. No soft label: the annotators answered survey items, not hate levels.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.mhs"""
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from builder.schema import Record

URL = "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/measuring-hate-speech.parquet"
CACHE = Path("data/sources/mhs/measuring-hate-speech.parquet")
LICENCE = "CC-BY-4.0"
CITATION = ("Sachdeva, P., Barreto, R., Bacon, G., Sahn, A., von Vacano, C. and Kennedy, C. J. (2022). The Measuring "
            "Hate Speech Corpus: Leveraging Rasch Measurement Theory for Data Perspectivism. NLPerspectives @ LREC "
            "2022. https://aclanthology.org/2022.nlperspectives-1.11; Kennedy, C. J., Bacon, G., Sahn, A. and von "
            "Vacano, C. (2020). Measuring a Hate Speech Spectrum with Faceted Rasch Item Response Theory and "
            "Perspective-Aware, Explainable-by-Design Deep Learning. https://arxiv.org/abs/2009.10277")
PLATFORMS = {0: "reddit", 1: "unknown", 2: "twitter", 3: "youtube"}  # 1: the 70 reference comments, mixed platforms
CATEGORIES = {"target_race": "race_ethnicity", "target_religion": "religion", "target_origin": "national_origin",
              "target_gender": "gender", "target_sexuality": "sexual_orientation", "target_age": "age",
              "target_disability": "disability"}
GROUPS = {
    "target_race_asian": "asian", "target_race_black": "black", "target_race_latinx": "latinx",
    "target_race_middle_eastern": "middle_eastern", "target_race_native_american": "indigenous",
    "target_race_pacific_islander": "pacific_islander", "target_race_white": "white",
    "target_religion_atheist": "nonreligious", "target_religion_buddhist": "buddhist",
    "target_religion_christian": "christian", "target_religion_hindu": "hindu", "target_religion_jewish": "jewish",
    "target_religion_mormon": "mormon", "target_religion_muslim": "muslim",
    "target_origin_immigrant": "immigrants", "target_origin_migrant_worker": "migrant_workers",
    "target_origin_undocumented": "undocumented_immigrants",
    "target_gender_men": "men", "target_gender_women": "women", "target_gender_non_binary": "non_binary",
    "target_gender_transgender_men": "transgender_men", "target_gender_transgender_women": "transgender_women",
    "target_gender_transgender_unspecified": "transgender",
    "target_sexuality_bisexual": "bisexual", "target_sexuality_gay": "gay", "target_sexuality_lesbian": "lesbian",
    "target_sexuality_straight": "heterosexual",
    "target_age_children": "children", "target_age_teenagers": "teenagers", "target_age_young_adults": "young_adults",
    "target_age_middle_aged": "middle_aged", "target_age_seniors": "seniors",
    "target_disability_physical": "physical_disability", "target_disability_cognitive": "cognitive_disability",
    "target_disability_neurological": "neurological_disability",
    "target_disability_visually_impaired": "visually_impaired",
    "target_disability_hearing_impaired": "hearing_impaired", "target_disability_unspecific": "disabled",
}


def fetch(cache: Path = CACHE) -> dict[int, list[dict]]:
    """The Parquet file, downloaded once, grouped into one list of annotation rows per comment ID."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    posts = defaultdict(list)
    columns = ["comment_id", "annotator_id", "platform", "hate_speech_score", "text", *CATEGORIES, *GROUPS]
    for row in pq.read_table(cache, columns=columns).to_pylist():
        posts[row["comment_id"]].append(row)
    return posts


def level(score: float) -> str:
    return "hateful" if score > 0.5 else "neither" if score < -1 else "unknown"


def majority(rows: list[dict], column: str) -> bool:
    return sum(r[column] for r in rows) * 2 > len(rows)


def records(posts: dict[int, list[dict]]) -> list[Record]:
    out = []
    for comment_id, rows in posts.items():
        first = rows[0]
        out.append(Record(
            source="mhs", source_id=str(comment_id), platform=PLATFORMS[first["platform"]], text=first["text"],
            text_form="raw", hate=level(first["hate_speech_score"]),
            label_raw=f"hate_speech_score={first['hate_speech_score']}",
            target_category=tuple(sorted({c for col, c in CATEGORIES.items() if majority(rows, col)})),
            target_group=tuple(sorted({g for col, g in GROUPS.items() if majority(rows, col)})),
            n_annotators=len(rows), soft_label=None, licence=LICENCE))
    return out


def check(posts: dict[int, list[dict]], out: list[Record]) -> None:
    """Fails unless the counts match the dataset card, each comment has a single score, text and platform, and the
    reference set is the 70 comments the authors describe; then prints the totals."""
    rows = [r for v in posts.values() for r in v]
    if (counts := (len(posts), len(rows), len({r["annotator_id"] for r in rows}))) != (39565, 135556, 7912):
        raise ValueError(f"comments, annotations, annotators: {counts}; the card says (39565, 135556, 7912)")
    if any(len({(r["hate_speech_score"], r["text"], r["platform"]) for r in v}) > 1 for v in posts.values()):
        raise ValueError("a comment has more than one score, text or platform")
    if (reference := sum(v[0]["platform"] == 1 for v in posts.values())) != 70:
        raise ValueError(f"{reference} reference comments, the authors describe 70")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    posts = fetch()
    out = records(posts)
    check(posts, out)
    return out


if __name__ == "__main__":
    build()
