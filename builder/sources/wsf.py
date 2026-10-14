# builder/sources/wsf.py
"""White Supremacy Forum (de Gibert et al., 2018): 10,944 sentences from Stormfront posts by 2,792 users, labelled hate,
no hate, relation (hateful only together with other sentences of the post) or idk/skip (not English, or too little to
judge), from the authors' GitHub (CC BY-SA 3.0 Spain). Hate is hateful; no hate is not hateful, since the dataset does
not separate offensive from neutral; relation and idk/skip are unknown for a sentence read alone. No targets. The
text is the authors' tokenised form, with spaces around punctuation. Users are pseudonymised into author ids.

Run from the repository root: uv run --isolated --only-group builder python -m builder.sources.wsf"""
import csv
import io
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from builder.privacy import pseudonym
from builder.schema import Record

URL = "https://github.com/Vicomtech/hate-speech-dataset/archive/refs/heads/master.zip"
CACHE = Path("data/sources/wsf/hate-speech-dataset-master.zip")
ROOT = "hate-speech-dataset-master/"
LICENCE = "CC-BY-SA-3.0-ES"
CITATION = ("de Gibert, O., Perez, N., García-Pablos, A. and Cuadros, M. (2018). Hate Speech Dataset from a White "
            "Supremacy Forum. Second Workshop on Abusive Language Online. https://aclanthology.org/W18-5102")
LEVELS = {"hate": "hateful", "noHate": "not_hateful", "relation": "unknown", "idk/skip": "unknown"}


def fetch(cache: Path = CACHE) -> tuple[list[dict], dict[str, str]]:
    """The annotation metadata and every sentence by file id, from the repository archive, downloaded once."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    with zipfile.ZipFile(cache) as z:
        with io.TextIOWrapper(z.open(ROOT + "annotations_metadata.csv"), encoding="utf-8", newline="") as f:
            meta = list(csv.DictReader(f))
        texts = {n.split("/")[-1].removesuffix(".txt"): z.read(n).decode("utf-8")
                 for n in z.namelist() if n.startswith(ROOT + "all_files/") and n.endswith(".txt")}
    return meta, texts


def records(meta: list[dict], texts: dict[str, str]) -> list[Record]:
    return [Record(
        source="wsf", source_id=r["file_id"], platform="stormfront", text=texts[r["file_id"]], text_form="preprocessed",
        hate=LEVELS[r["label"]], label_raw=f"{r['label']} contexts={r['num_contexts']}", target_category=(),
        target_group=(), n_annotators=None, soft_label=None, licence=LICENCE,
        author_id=pseudonym("stormfront", r["user_id"])) for r in meta]


def check(meta: list[dict], texts: dict[str, str], out: list[Record]) -> None:
    """Fails unless the metadata lists the 10,944 sentences of the authors' dataset card, each once, every one with its
    sentence file and no file left over, and every label is known; then prints the totals."""
    ids = [r["file_id"] for r in meta]
    if len(ids) != 10944 or len(set(ids)) != len(ids) or set(ids) != texts.keys():
        raise ValueError(f"{len(ids)} labelled sentences, {len(texts)} files; the card gives 10,944 of each, matched")
    if missing := {r["label"] for r in meta} - LEVELS.keys():
        raise ValueError(f"labels with no mapping: {missing}")
    print("hate levels:", dict(Counter(r.hate for r in out)))


def build() -> list[Record]:
    meta, texts = fetch()
    out = records(meta, texts)
    check(meta, texts, out)
    return out


if __name__ == "__main__":
    build()
