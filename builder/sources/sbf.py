# builder/sources/sbf.py
"""Social Bias Frames (Sap et al., 2020): 44,671 posts from Reddit, Twitter, Gab and Stormfront with 147,139 annotation
rows, from the authors' site (CC BY 4.0). SBF asks whether a post could offend anyone and whether it targets a group or
an individual, never whether it is hate. So a post is hateful when a majority find it offensive and a majority of those
asked say it targets a group, offensive when they say an individual, neither when a majority find it inoffensive, and
unknown on a tie; a "maybe" abstains, which makes the offensive vote the authors' mean-above-0.5 rule. Targets are
kept on hateful posts, by majority of the annotators who named a group. A post is its text, as in the authors' own
aggregation: one text can sit under several HIT IDs, and 203 texts sit in more than one of the authors' splits. Two
origins reached SBF tokenised, so their texts count as preprocessed: Stormfront (the White Supremacy Forum corpus) and
r/meanjokes, also lower-cased with tags such as <number> and <repeat>.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.sbf"""
import csv
import io
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from builder.schema import SOFT_LEVELS, Record

URL = "https://maartensap.com/social-bias-frames/SBIC.v2.tgz"
CACHE = Path("data/sources/sbf/SBIC.v2.tgz")
LICENCE = "CC-BY-4.0"
CITATION = ("Sap, M., Gabriel, S., Qin, L., Jurafsky, D., Smith, N. A. and Choi, Y. (2020). Social Bias Frames: "
            "Reasoning about Social and Power Implications of Language. ACL 2020. "
            "https://aclanthology.org/2020.acl-main.486")
SPLITS = ("trn", "dev", "tst")
VOTE = {"1.0": 1, "0.0": -1}  # "0.5" (maybe) and "" (not asked) abstain
PREPROCESSED = ("Stormfront", "r/meanjokes")
COARSE = {"race": ("race_ethnicity",), "gender": ("gender",), "disabled": ("disability",), "social": ("other",),
          "body": ("other",), "victim": ("other",), "culture": ()}  # culture mixes religion and origin: groups decide
GROUPS = {  # the annotators' commonest target strings, lower-cased: (our target categories, our target group)
    "black folks": (("race_ethnicity",), "black"), "asian folks": (("race_ethnicity",), "asian"),
    "latino/latina folks": (("race_ethnicity",), "latinx"),
    "native american/first nation folks": (("race_ethnicity",), "indigenous"),
    "white folks": (("race_ethnicity",), "white"), "white people": (("race_ethnicity",), "white"),
    "people of color": (("race_ethnicity",), "non_white"), "non-whites": (("race_ethnicity",), "non_white"),
    "arabic folks": (("race_ethnicity",), "arab"), "jewish folks": (("race_ethnicity", "religion"), "jewish"),
    "muslim folks": (("religion",), "muslim"), "christian folks": (("religion",), "christian"),
    "immigrants": (("national_origin",), "immigrants"), "africans": (("national_origin",), "african"),
    "african folks": (("national_origin",), "african"), "ethiopians": (("national_origin",), "ethiopian"),
    "ethiopian folks": (("national_origin",), "ethiopian"),
    "women": (("gender",), "women"), "men": (("gender",), "men"), "trans women": (("gender",), "transgender_women"),
    "trans men": (("gender",), "transgender_men"), "gay men": (("sexual_orientation",), "gay"),
    "lesbian women": (("sexual_orientation",), "lesbian"), "bisexual women": (("sexual_orientation",), "bisexual"),
    "bisexual men": (("sexual_orientation",), "bisexual"),
    "physically disabled folks": (("disability",), "physical_disability"),
    "mentally disabled folks": (("disability",), "cognitive_disability"),
    "folks with mental illness/disorder": (("disability",), "mental_illness"),
    "folks with physical illness/disorder": (("disability",), "physical_illness"),
    "autistic folks": (("disability",), "autistic"),
    "liberals": (("political",), "liberals"), "conservatives": (("political",), "conservatives"),
    "feminists": (("political",), "feminists"), "feminist": (("political",), "feminists"),
    "poor folks": (("other",), "poor"), "overweight/fat folks": (("other",), "fat"),
    "children": (("age",), "children"), "kids": (("age",), "children"), "old folks": (("age",), "seniors"),
    "seniors": (("age",), "seniors"),
}
TABLE_2 = {  # the paper's posts per origin: it prints 2,011 for Microaggressions, but its total of 44,671 needs 2,010
    ("r/darkjokes",): 10095, ("r/meanjokes",): 3483, ("r/offensivejokes", "r/offensivequestions"): 356,
    ("redditMicroagressions",): 2010, ("t/founta",): 11864, ("t/davidson",): 3008, ("t/waseem",): 1816,
    ("Gab",): 3715, ("Stormfront",): 4016, ("r/Incels", "r/MensRights"): 4308,
}


def fetch(cache: Path = CACHE) -> dict[str, list[dict]]:
    """Every annotation row from the three split files, downloaded once, grouped into one list per post text."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    csv.field_size_limit(sys.maxsize)
    posts = defaultdict(list)
    with tarfile.open(cache) as t:
        for split in SPLITS:
            for row in csv.DictReader(io.TextIOWrapper(t.extractfile(f"SBIC.v2.{split}.csv"), encoding="utf-8")):
                posts[row["post"]].append(row)
    return posts


def annotators(rows: list[dict]) -> list[list[dict]]:
    """One list of rows per annotation: an annotator writes one row per group they name."""
    by = defaultdict(list)
    for r in rows:
        by[r["HITId"], r["WorkerId"]].append(r)
    return list(by.values())


def named(rows: list[dict]) -> tuple[set[str], set[str]]:
    """The target categories and groups one annotator named; a group string with no mapping keeps its row's coarse
    category."""
    categories, groups = set(), set()
    for r in rows:
        strings = [s.strip().lower() for s in r["targetMinority"].split(",") if s.strip()]
        for s in strings:
            if s in GROUPS:
                categories |= set(GROUPS[s][0])
                groups.add(GROUPS[s][1])
        if not strings or any(s not in GROUPS for s in strings):
            categories |= set(COARSE.get(r["targetCategory"], ()))
    return categories, groups


def level(person: list[dict]) -> str | None:
    off, who = person[0]["offensiveYN"], person[0]["whoTarget"]
    return "neither" if off == "0.0" else ("hateful" if who == "1.0" else "offensive") if off == "1.0" else None


def aggregate(rows: list[dict]) -> tuple[str, tuple[str, ...], tuple[str, ...], dict[str, float] | None]:
    """One post's rows combined into our hate level, the target categories and groups kept, and the soft label over
    the annotators who answered yes or no."""
    people = annotators(rows)
    offensive = sum(VOTE.get(p[0]["offensiveYN"], 0) for p in people)
    group = sum(VOTE.get(p[0]["whoTarget"], 0) for p in people)
    hate = ("neither" if offensive < 0 else "unknown" if offensive == 0 or group == 0
            else "hateful" if group > 0 else "offensive")
    asked = [named(p) for p in people if p[0]["whoTarget"] == "1.0"] if hate == "hateful" else []
    categories = Counter(c for cats, _ in asked for c in cats)
    groups = Counter(g for _, gs in asked for g in gs)
    decided = [lv for p in people if (lv := level(p))]
    soft = {lv: decided.count(lv) / len(decided) for lv in SOFT_LEVELS} if decided else None
    return (hate, tuple(sorted(c for c, n in categories.items() if n * 2 > len(asked))),
            tuple(sorted(g for g, n in groups.items() if n * 2 > len(asked))), soft)


def records(posts: dict[str, list[dict]]) -> list[Record]:
    out = []
    for text, rows in posts.items():
        hate, categories, groups, soft = aggregate(rows)
        source = rows[0]["dataSource"]
        people = annotators(rows)
        out.append(Record(
            source="sbf", source_id=min(r["HITId"] for r in rows), text=text,
            text_form="preprocessed" if source in PREPROCESSED else "raw", hate=hate,
            platform="twitter" if source.startswith("t/") else source.lower() if source in ("Gab", "Stormfront")
            else "reddit",
            label_raw="|".join(f"offensive={p[0]['offensiveYN']} group={p[0]['whoTarget']}" for p in people),
            target_category=categories, target_group=groups, n_annotators=len(people), soft_label=soft,
            licence=LICENCE))
    return out


def check(posts: dict[str, list[dict]], out: list[Record]) -> None:
    """Fails unless the posts and annotation rows number the paper's 44,671 and 147,139, its posts per origin are
    reproduced, and every target category the annotators used has a mapping; then prints the totals."""
    rows = [r for v in posts.values() for r in v]
    if (counts := (len(posts), len(rows))) != (44671, 147139):
        raise ValueError(f"posts, annotation rows: {counts}; the paper says (44671, 147139)")
    origins = Counter(v[0]["dataSource"] for v in posts.values())
    if built := {k: n for k, n in TABLE_2.items() if sum(origins[s] for s in k) != n}:
        raise ValueError(f"posts per origin differ from Table 2 for {built}: built {dict(origins)}")
    if missing := {r["targetCategory"] for r in rows} - COARSE.keys() - {""}:
        raise ValueError(f"target categories with no mapping: {missing}")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    posts = fetch()
    out = records(posts)
    check(posts, out)
    return out


if __name__ == "__main__":
    build()
