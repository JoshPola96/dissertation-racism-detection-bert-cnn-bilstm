# builder/sources/imsypp.py
"""IMSyPP English YouTube corpus (2021): 62,414 comments on Covid-19 videos, from CLARIN.SI (CC BY-SA 4.0), labelled
appropriate, inappropriate, offensive or violent, with a target for offensive and violent ones. The 51,655 training
comments have two annotations each; the 10,759 evaluation comments four, two made with the conversation's context and
two without. Per annotation, offensive or violent speech at a protected group (racism, migrants, islamophobia,
antisemitism, religion, sexism, homophobia) is hateful, at politics, media, an individual or others offensive, and
inappropriate (vulgar, untargeted) speech offensive. A comment takes the strict majority of its annotations, unknown on
a tie; an annotation without a type abstains. Targets are kept by majority of the annotators who gave one. The context
file marks replies with a "-*-*-*- " prefix, removed here. A spreadsheet turned comments starting with "+" into
"#ERROR! (Formula parse error.)" in some files: annotations of that error abstain, the comment's real text is used where
another file kept it, and the 4 comments that survive only as the error are left out. Where versions of a comment
differ (13, by truncation or editing), the longest is kept.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.imsypp"""
import csv
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from builder.schema import SOFT_LEVELS, Record

URL = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1454/"
CACHE = Path("data/sources/imsypp")
FILES = ("IMSyPP_EN_YouTube_comments_train.csv", "IMSyPP_EN_YouTube_comments_evaluation_context.csv",
         "IMSyPP_EN_YouTube_comments_evaluation_no_context.csv")
LICENCE = "CC-BY-SA-4.0"
CITATION = ("Ljubešić, N., Mozetič, I., Cinelli, M. and Kralj Novak, P. (2021). English YouTube Hate Speech Corpus. "
            "CLARIN.SI. http://hdl.handle.net/11356/1454")
REPLY = "-*-*-*- "
ERROR = "#ERROR! (Formula parse error.)"
TARGETS = {  # target label: (our target categories, our target group, a protected group?)
    "1. racism": (("race_ethnicity",), None, True), "2. migrants": (("national_origin",), "immigrants", True),
    "3. islamophobia": (("religion",), "muslim", True),
    "4. antisemitism": (("race_ethnicity", "religion"), "jewish", True),
    "5. religion": (("religion",), None, True), "6. homophobia": (("sexual_orientation",), None, True),
    "7. sexism": (("gender",), None, True), "8. ideology": (("political",), None, False),
    "9. media": (("other",), None, False), "10. politics": (("political",), None, False),
    "11. individual": ((), None, False), "12. other": (("other",), None, False),
}


def fetch(cache: Path = CACHE) -> dict[str, list[dict]]:
    """Every annotation from the three files, downloaded once, grouped into one list per comment."""
    csv.field_size_limit(sys.maxsize)
    comments = defaultdict(list)
    for name in FILES:
        if not (cache / name).exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL + name, cache / name)
        with (cache / name).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                comments[row["Comment_ID"]].append({**row, "file": name})
    return comments


def text(annotation: dict) -> str:
    return annotation["Text"].removeprefix(REPLY).strip()


def level(annotation: dict) -> str | None:
    kind, target = annotation["Type"], annotation["Target"]
    if text(annotation) == ERROR:
        return None
    if kind.startswith("0."):
        return "neither"
    if kind.startswith("1."):
        return "offensive"
    if kind.startswith(("2.", "3.")):
        return "hateful" if target in TARGETS and TARGETS[target][2] else "offensive"
    return None


def records(comments: dict[str, list[dict]]) -> list[Record]:
    out = []
    for comment_id, rows in comments.items():
        versions = sorted({text(r) for r in rows} - {ERROR}, key=lambda t: (-len(t), t))
        if not versions:
            continue
        decided = [lv for r in rows if (lv := level(r))]
        votes = Counter(decided)
        hate = next((lv for lv, n in votes.items() if n * 2 > len(decided)), "unknown")
        asked = [r["Target"] for r in rows if r["Target"] in TARGETS] if hate in ("hateful", "offensive") else []
        kept = [TARGETS[t] for t, n in Counter(asked).items() if n * 2 > len(asked)]
        out.append(Record(
            source="imsypp", source_id=comment_id, platform="youtube", text=versions[0], text_form="raw", hate=hate,
            label_raw="|".join(f"{r['Type'] or '-'}/{r['Target'] or '-'}" for r in rows),
            target_category=tuple(sorted({c for cats, _, _ in kept for c in cats})),
            target_group=tuple(sorted({g for _, g, _ in kept if g})), n_annotators=len(decided),
            soft_label={lv: votes[lv] / len(decided) for lv in SOFT_LEVELS} if decided else None, licence=LICENCE))
    return out


def check(comments: dict[str, list[dict]], out: list[Record]) -> None:
    """Fails unless the files hold the 51,655 training and 10,759 evaluation comments of the corpus record, every
    evaluation comment sits in both evaluation files, every version of a comment opens the same way (so no id joins
    two comments), and every label is known; then prints the totals."""
    files = {name: {c for c, rows in comments.items() if any(r["file"] == name for r in rows)} for name in FILES}
    if (len(files[FILES[0]]), len(files[FILES[1]])) != (51655, 10759) or files[FILES[1]] != files[FILES[2]]:
        raise ValueError("the training and evaluation comments differ from the corpus record")
    if any(len({t[:10] for t in {text(r) for r in rows} - {ERROR}}) > 1 for rows in comments.values()):
        raise ValueError("an id holds versions that open differently: two comments may share it")
    if unknown := {r["Target"] for rows in comments.values() for r in rows} - TARGETS.keys() - {""}:
        raise ValueError(f"targets with no mapping: {unknown}")
    print("hate levels:", dict(Counter(r.hate for r in out)))
    print("target categories:", dict(Counter(c for r in out for c in r.target_category)))


def build() -> list[Record]:
    comments = fetch()
    out = records(comments)
    check(comments, out)
    return out


if __name__ == "__main__":
    build()
