# builder/sources/cad.py
"""Contextual Abuse Dataset (Vidgen et al., 2021): 26,550 Reddit entries with 27,494 expert-adjudicated labels, one row
per label, from Zenodo (CC BY 4.0). Keeps the 23,417 entries the authors used, leaving out those they excluded as
empty, bot, non-English or image-only. CAD's taxonomy sets the level: identity-directed abuse is hateful, abuse of an
affiliation (a party, a profession) or of a person is offensive, neutral and counter speech are neither, and non-hateful
use of a slur, which spans reclaimed use and quotation that keeps the slur's force, is not hateful, as in the Slur
Corpus; an entry with several labels takes the most severe. Targets are the annotators' free text, so only the common
ones map to groups. Usernames become keyed pseudonyms, which pseudonymise rather than anonymise them. No vote counts or
soft label: disagreements were adjudicated.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.cad"""
import csv
import io
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from builder.privacy import pseudonym
from builder.schema import Record

URL = "https://zenodo.org/api/records/4881008/files/data.zip/content"
CACHE = Path("data/sources/cad/data.zip")
MEMBER = "data/cad_v1_1.tsv"
LICENCE = "CC-BY-4.0"
CITATION = ("Vidgen, B., Nguyen, D., Margetts, H., Rossini, P. and Tromble, R. (2021). Introducing CAD: the "
            "Contextual Abuse Dataset. NAACL-HLT 2021. https://aclanthology.org/2021.naacl-main.182")
LEVELS = {"IdentityDirectedAbuse": "hateful", "AffiliationDirectedAbuse": "offensive",
          "PersonDirectedAbuse": "offensive", "CounterSpeech": "neither", "Slur": "not_hateful", "Neutral": "neither"}
SEVERITY = ("hateful", "offensive", "not_hateful", "neither")
CATEGORIES = {"race": "race_ethnicity", "ethnicity": "race_ethnicity", "religion": "religion",
              "nationality/origin": "national_origin", "gender": "gender", "sexuality": "sexual_orientation",
              "ableness/disability": "disability", "age": "age", "political affiliation": "political", "class": "other",
              "profession/occupation": "other", "perceived negative groups": "other", "hateful entities": "other",
              "other": "other"}
GROUPS = {  # the annotators' commonest target strings
    "black people": ("black",), "black men": ("black", "men"), "white people": ("white",),
    "white men": ("white", "men"), "white women": ("white", "women"), "non-white people": ("non_white",),
    "chinese women": ("chinese", "women"), "people from china": ("chinese",), "people from pakistan": ("pakistani",),
    "people from america": ("american",), "people from mexico": ("mexican",), "people from india": ("indian",),
    "people from africa": ("african",), "people from asia": ("asian",), "people from britain": ("british",),
    "ethnic minorities": ("ethnic_minorities",), "romani people": ("romani",), "latinx": ("latinx",),
    "arabs": ("arab",), "mixed race/ethnicity": ("mixed_race",), "indigenous people": ("indigenous",),
    "muslims": ("muslim",), "jewish people": ("jewish",), "christians": ("christian",), "catholics": ("christian",),
    "hindus": ("hindu",), "atheists": ("nonreligious",),
    "women": ("women",), "men": ("men",), "gay men": ("gay", "men"), "gay people": ("gay",),
    "transgender people": ("transgender",), "non-gender dysphoric transgender people": ("transgender",),
    "sexual and gender minorities": ("lgbtq",), "lgbtqa community": ("lgbtq",),
    "people with mental disabilities": ("cognitive_disability",), "people with autism": ("autistic",),
    "illegal immigrants": ("undocumented_immigrants",), "working class people": ("working_class",),
    "elderly people": ("seniors",),
}
KEPT = ("train", "dev", "test")


def fetch(cache: Path = CACHE) -> dict[str, list[dict]]:
    """Every label row from the zip, downloaded once, grouped into one list per entry."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    csv.field_size_limit(sys.maxsize)
    entries = defaultdict(list)
    with zipfile.ZipFile(cache) as z, io.TextIOWrapper(z.open(MEMBER), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            entries[row["info_id"]].append(row)
    return entries


def records(entries: dict[str, list[dict]]) -> list[Record]:
    out = []
    for entry_id, rows in entries.items():
        first = rows[0]
        if first["split"] not in KEPT:
            continue
        targets = [r["annotation_Target"] for r in rows]
        parts = {p.strip() for r in rows for p in r["annotation_Target_top.level.category"].split(",")} - {'"NA"'}
        author = first["meta_author"]
        out.append(Record(
            source="cad", source_id=entry_id, platform="reddit", text=first["meta_text"].replace("[linebreak]", "\n"),
            text_form="raw", hate=min((LEVELS[r["annotation_Primary"]] for r in rows), key=SEVERITY.index),
            label_raw="|".join(r["annotation_Primary"] for r in rows),
            target_category=tuple(sorted({CATEGORIES[p] for p in parts})),
            target_group=tuple(sorted({g for t in targets for g in GROUPS.get(t, ())})),
            n_annotators=None, soft_label=None, licence=LICENCE,
            author_id=None if author == "[deleted]" else pseudonym("reddit", author)))
    return out


def check(entries: dict[str, list[dict]], out: list[Record]) -> None:
    """Fails unless the counts match the paper (26,550 entries, 27,494 labels, splits of 13,584, 4,526 and 5,307), every
    label and target category has a mapping, and each entry has one split, text and author; then prints the totals."""
    rows = [r for v in entries.values() for r in v]
    if (counts := (len(entries), len(rows))) != (26550, 27494):
        raise ValueError(f"entries, labels: {counts}; the paper says (26550, 27494)")
    splits = Counter(v[0]["split"] for v in entries.values())
    if [splits[s] for s in KEPT] != [13584, 4526, 5307]:
        raise ValueError(f"splits {dict(splits)}; the paper says train 13,584, dev 4,526, test 5,307")
    if missing := {r["annotation_Primary"] for r in rows} - LEVELS.keys():
        raise ValueError(f"labels with no mapping: {missing}")
    parts = {p.strip() for r in rows for p in r["annotation_Target_top.level.category"].split(",")} - {'"NA"'}
    if missing := parts - CATEGORIES.keys():
        raise ValueError(f"target categories with no mapping: {missing}")
    if any(len({(r["split"], r["meta_text"], r["meta_author"]) for r in v}) > 1 for v in entries.values()):
        raise ValueError("an entry has more than one split, text or author")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    entries = fetch()
    out = records(entries)
    check(entries, out)
    return out


if __name__ == "__main__":
    build()
