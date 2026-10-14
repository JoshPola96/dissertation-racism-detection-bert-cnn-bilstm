# builder/sources/implicit_hate.py
"""Implicit Hate (ElSherief et al., 2021): 21,480 posts, tweets from US extremist accounts plus 548 taken from Social
Bias Frames, each labelled explicit hate, implicit hate or not hate, with a free-text target for the implicit ones
(CC BY 4.0). The authors ask users to complete a survey before downloading, so the corpus zip is fetched by hand. Both
kinds of hate are hateful, with that explicitness; not hate is not hateful, since the corpus does not separate
offensive from neutral. The commonest target strings map to groups. The text is the authors' lower-cased,
pre-processed form. The README inside the zip counts 22,056 tweets, 576 more than its files hold.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.implicit_hate"""
import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

from builder.schema import Record

SURVEY = "https://forms.gle/QxCpEbVp91Z35hWFA"
CACHE = Path("data/sources/implicit_hate/implicit-hate-corpus-nov-2021.zip")
FOLDER = "implicit-hate-corpus/"
LICENCE = "CC-BY-4.0"
CITATION = ("ElSherief, M., Ziems, C., Muchlinski, D., Anupindi, V., Seybolt, J., De Choudhury, M. and Yang, D. "
            "(2021). Latent Hatred: A Benchmark for Understanding Implicit Hate Speech. EMNLP 2021. "
            "https://aclanthology.org/2021.emnlp-main.29")
LEVELS = {"explicit_hate": ("hateful", "explicit"), "implicit_hate": ("hateful", "implicit"),
          "not_hate": ("not_hateful", "unknown")}
GROUPS = {  # the commonest target strings, lower-cased: (our target categories, our target groups)
    "immigrants": (("national_origin",), ("immigrants",)), "refugees": (("national_origin",), ("refugees",)),
    "illegal immigrants": (("national_origin",), ("undocumented_immigrants",)),
    "illegals": (("national_origin",), ("undocumented_immigrants",)),
    "illegal aliens": (("national_origin",), ("undocumented_immigrants",)),
    "mexicans": (("national_origin",), ("mexican",)), "germans": (("national_origin",), ("german",)),
    "africans": (("national_origin",), ("african",)), "indians": (("national_origin",), ("indian",)),
    "white people": (("race_ethnicity",), ("white",)), "whites": (("race_ethnicity",), ("white",)),
    "white folks": (("race_ethnicity",), ("white",)), "white men": (("race_ethnicity", "gender"), ("white", "men")),
    "white women": (("race_ethnicity", "gender"), ("white", "women")),
    "black people": (("race_ethnicity",), ("black",)), "blacks": (("race_ethnicity",), ("black",)),
    "black folks": (("race_ethnicity",), ("black",)), "minorities": (("race_ethnicity",), ("ethnic_minorities",)),
    "non-white people": (("race_ethnicity",), ("non_white",)), "non-whites": (("race_ethnicity",), ("non_white",)),
    "people of color": (("race_ethnicity",), ("non_white",)), "arabs": (("race_ethnicity",), ("arab",)),
    "jews": (("race_ethnicity", "religion"), ("jewish",)),
    "jewish people": (("race_ethnicity", "religion"), ("jewish",)),
    "muslims": (("religion",), ("muslim",)), "islamic people": (("religion",), ("muslim",)),
    "christians": (("religion",), ("christian",)), "gays": (("sexual_orientation",), ("gay",)),
    "women": (("gender",), ("women",)),
    "liberals": (("political",), ("liberals",)), "progressives": (("political",), ("liberals",)),
    "leftists": (("political",), ("liberals",)),
    "white liberals": (("political", "race_ethnicity"), ("liberals", "white")),
    "democrats": (("political",), ("democrats",)), "conservatives": (("political",), ("conservatives",)),
    "white conservatives": (("political", "race_ethnicity"), ("conservatives", "white")),
    "conservative males": (("political", "gender"), ("conservatives", "men")), "antifa": (("political",), ("antifa",)),
}


def read(z: zipfile.ZipFile, name: str) -> list[dict]:
    with io.TextIOWrapper(z.open(FOLDER + name), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fetch(cache: Path = CACHE) -> tuple[list[dict], dict[str, str]]:
    """The stage-1 posts, each with its ID and its text (two files that line up row for row), and the stage-3 target
    of each implicit post by ID."""
    if not cache.exists():
        raise FileNotFoundError(f"{cache}: complete the authors' survey at {SURVEY}, then save their zip there")
    with zipfile.ZipFile(cache) as z:
        ids, posts = read(z, "implicit_hate_v1_stg1.tsv"), read(z, "implicit_hate_v1_stg1_posts.tsv")
        targets = {r["ID"]: r["target"] for r in read(z, "implicit_hate_v1_stg3.tsv") if r["ID"]}
    return [{**i, "post": p["post"], "post_class": p["class"]} for i, p in zip(ids, posts, strict=True)], targets


def records(rows: list[dict], targets: dict[str, str]) -> list[Record]:
    out = []
    for r in rows:
        hate, explicitness = LEVELS[r["class"]]
        categories, groups = GROUPS.get(targets.get(r["ID"], "").strip().lower(), ((), ()))
        out.append(Record(
            source="implicit_hate", source_id=r["ID"], platform="unknown" if r["ID"].startswith("SAP_") else "twitter",
            text=r["post"], text_form="preprocessed", hate=hate, label_raw=r["class"],
            target_category=tuple(sorted(categories)), target_group=tuple(sorted(groups)), n_annotators=None,
            soft_label=None, licence=LICENCE, explicitness=explicitness))
    return out


def check(rows: list[dict], targets: dict[str, str], out: list[Record]) -> None:
    """Fails unless the ID and text files agree row for row, the posts number 21,480 with the paper's 13,291 not
    hateful, and every target belongs to a stage-1 post; then prints the totals."""
    if any(r["class"] != r["post_class"] for r in rows):
        raise ValueError("the ID file and the text file disagree on a class: they no longer line up")
    if (counts := (len(rows), sum(r["class"] == "not_hate" for r in rows))) != (21480, 13291):
        raise ValueError(f"posts, not hate: {counts}; expected (21480, 13291)")
    if stray := targets.keys() - {r["ID"] for r in rows}:
        raise ValueError(f"{len(stray)} targets belong to no stage-1 post")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    rows, targets = fetch()
    out = records(rows, targets)
    check(rows, targets, out)
    return out


if __name__ == "__main__":
    build()
