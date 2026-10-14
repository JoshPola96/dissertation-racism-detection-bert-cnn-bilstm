# builder/sources/hateval.py
"""HatEval (Basile et al., 2019), English part: 12,971 tweets collected about immigrants or women, labelled for hate,
for whether hate targets a group or an individual, and for aggression, from the organisers' gated Hugging Face dataset
(CC BY-NC 4.0). Hate is hateful; the rest is not hateful, since HatEval asks only about hate. Each tweet keeps the
group it was collected about as its target. The release's test set holds 2,971 English tweets where the paper
describes 3,000.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.hateval"""
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from builder.schema import Record

REPO = "valeriobasile/HatEval"
CACHE = Path("data/sources/hateval")
SPLITS = ("train", "dev", "test")
LICENCE = "CC-BY-NC-4.0"
CITATION = ("Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F. M., Rosso, P. and "
            "Sanguinetti, M. (2019). SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants "
            "and Women in Twitter. SemEval 2019. https://aclanthology.org/S19-2007")
TARGETS = {"mig": ("national_origin", "immigrants"), "mis": ("gender", "women")}
TABLE_1 = {  # the paper's English training set (train and dev), per target: % hateful; % of hate at an individual,
    "mig": (39.76, 5.89, 55.08),  # and % of hate that is aggressive
    "mis": (44.44, 64.94, 30.06),
}


def fetch(cache: Path = CACHE) -> list[dict]:
    """The English tweets of the three splits, from a download made with the user's own Hugging Face access."""
    paths = [cache / "data" / f"{s}-00000-of-00001.parquet" for s in SPLITS]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"{cache}: accept the terms of the Hugging Face dataset {REPO}, log in, then run "
                                f"hf download {REPO} --repo-type dataset --local-dir {cache}")
    return [{**r, "split": s} for s, p in zip(SPLITS, paths, strict=True)
            for r in pq.read_table(p).to_pylist() if r["language"] == "en"]


def records(rows: list[dict]) -> list[Record]:
    return [Record(
        source="hateval", source_id=str(r["id"]), platform="twitter", text=r["text"], text_form="raw",
        hate="hateful" if r["HS"] == 1 else "not_hateful",
        label_raw=f"HS={r['HS']} TR={r['TR']} AG={r['AG']} target={r['target']}",
        target_category=(TARGETS[r["target"]][0],), target_group=(TARGETS[r["target"]][1],), n_annotators=None,
        soft_label=None, licence=LICENCE) for r in rows]


def check(rows: list[dict], out: list[Record]) -> None:
    """Fails unless the training tweets reproduce the paper's Table 1 for each target and every tweet has its own id;
    then prints the totals."""
    for target, published in TABLE_1.items():
        part = [r for r in rows if r["target"] == target and r["split"] != "test"]
        hate = [r for r in part if r["HS"] == 1]
        built = tuple(round(100 * n / d, 2) for n, d in ((len(hate), len(part)),
                      (sum(r["TR"] == 1 for r in hate), len(hate)), (sum(r["AG"] == 1 for r in hate), len(hate))))
        if built != published:
            raise ValueError(f"{target}: {built}; the paper's Table 1 gives {published}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("a tweet id repeats")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    rows = fetch()
    out = records(rows)
    check(rows, out)
    return out


if __name__ == "__main__":
    build()
