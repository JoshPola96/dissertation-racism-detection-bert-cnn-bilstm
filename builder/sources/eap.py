# builder/sources/eap.py
"""East Asian Prejudice (Vidgen et al., 2020): 20,000 COVID-19-era tweets, each labelled by two annotators and settled
by an expert, from Zenodo (CC BY 4.0). The expert's label sets the level. Its "entity-directed hostility" covers East
Asian identities and institutions alike, so, as CAD separates them, hostility at people or places is hateful and
hostility only at institutions, office-holders or professions is offensive; criticism ("critical, but not abusive"),
counter speech, discussion of prejudice and none of the above are neither. Targets were asked only for hostility and
criticism. Usernames were anonymised by the authors, who also lower-cased about half the texts (58% hold no capital,
against 7% of Davidson's tweets, and 1,441 retweets begin "rt", which Twitter never writes), so the texts count as
preprocessed.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.eap"""
import csv
import io
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from builder.schema import SOFT_LEVELS, Record

URL = "https://zenodo.org/api/records/3816667/files/hs_AsianPrejudice_20kdataset_cleaned_anonymized.tsv/content"
CACHE = Path("data/sources/eap/hs_AsianPrejudice_20kdataset_cleaned_anonymized.tsv")
LICENCE = "CC-BY-4.0"
CITATION = ("Vidgen, B., Botelho, A., Broniatowski, D., Guest, E., Hall, M., Margetts, H., Tromble, R., Waseem, Z. "
            "and Hale, S. (2020). Detecting East Asian Prejudice on Social Media. Fourth Workshop on Online Abuse "
            "and Harms. https://aclanthology.org/2020.alw-1.19")
ENTITY = ("entity_directed_hostility", "entity_directed_criticism")
POLITICAL = ("government", "party", "communist", "state", "president", "emperor", "military", "politician", "police",
             "diplomat", "supporter", "sympathizer")
PROFESSIONS = ("media", "corporation", "scientist", "doctor", "healthcare", "restaurant", "market", "entertainer")
GROUPS = (("east asia", "east_asian"), ("asian american", "asian"), ("chinese", "chinese"), ("china", "chinese"),
          ("wuhan", "chinese"), ("japan", "japanese"), ("korea", "korean"), ("taiwan", "taiwanese"),
          ("hong kong", "hong_kong"), ("singapore", "singaporean"))
TABLE_1 = {  # the paper's tweets per category: it prints 13,528 neutral, but its total of 20,000 needs 13,524
    "entity_directed_hostility": 3898, "entity_directed_criticism": 1433, "counter_speech": 116,
    "discussion_of_eastasian_prejudice": 1029, "none_of_the_above": 13524,
}


def fetch(cache: Path = CACHE) -> list[dict]:
    """The 20,000-tweet file, downloaded once. It escapes a quote inside a field with a backslash but leaves
    backslashes themselves unescaped, so each escaped quote is doubled, the form the csv module reads; its own
    escapechar would also eat the backslash of a shrug."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    csv.field_size_limit(sys.maxsize)
    with cache.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(io.StringIO(f.read().replace('\\"', '""')), delimiter="\t"))


def targets(*fields: str) -> set[str]:
    """The lower-cased target strings in the annotators' free-text target fields."""
    return {p.strip().lower() for f in fields for p in f.replace(";", ",").split(",")} - {"", "na", "none"}


def kind(target: str) -> str:
    """Political for institutions, office-holders and affiliations, other for professions, else people or places."""
    return ("political" if any(w in target for w in POLITICAL) else "other" if any(w in target for w in PROFESSIONS)
            else "race_ethnicity")


def level(label: str, named: set[str]) -> str:
    if label != "entity_directed_hostility":
        return "neither"
    return "hateful" if not named or any(kind(t) == "race_ethnicity" for t in named) else "offensive"


def records(rows: list[dict]) -> list[Record]:
    out = []
    for r in rows:
        named = targets(r["target.annot1.clean"], r["target.annot2.clean"]) if r["expert"] in ENTITY else set()
        categories = {kind(t) for t in named}
        people = {t for t in named if kind(t) == "race_ethnicity"}
        votes = Counter(level(r[a], targets(r[f"target.{a}.clean"])) for a in ("annot1", "annot2") if r[a] != "NA")
        out.append(Record(
            source="eap", source_id=r["id"], platform="twitter", text=r["text"], text_form="preprocessed",
            hate=level(r["expert"], named), label_raw=f"expert={r['expert']} annot1={r['annot1']} annot2={r['annot2']}",
            target_category=tuple(sorted(categories)),
            target_group=tuple(sorted({g for t in people for key, g in GROUPS if key in t})),
            n_annotators=sum(votes.values()),
            soft_label={lv: votes[lv] / sum(votes.values()) for lv in SOFT_LEVELS},
            licence=LICENCE))
    return out


def check(rows: list[dict], out: list[Record]) -> None:
    """Fails unless the tweets per expert category reproduce the paper's Table 1, every tweet has one row and no
    escaped quote is left in a tweet; then prints the totals."""
    if (built := dict(Counter(r["expert"] for r in rows))) != TABLE_1:
        raise ValueError(f"tweets per category {built}; the paper's Table 1 gives {TABLE_1}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("a tweet appears on more than one row")
    if escaped := sum('\\"' in r.text for r in out):
        raise ValueError(f"{escaped} tweets still hold escaped quotes")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    rows = fetch()
    out = records(rows)
    check(rows, out)
    return out


if __name__ == "__main__":
    build()
