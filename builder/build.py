# builder/build.py
"""Builds HateAtlas: every source fetched, mapped and checked, then pooled, cleaned, redacted and de-duplicated into
Parquet tables of unique texts, each with a report beside it: one per licence tier, built only from sources whose
licences allow the tier's own, and one of every source for local research; then the dataset card for the published
tiers (builder/card.py). A source that cannot be fetched (a manual download not yet done, a host that does not answer)
is reported and left out; a source that fails its check stops the build, and so does a table that fails its quality
checks, before anything is written. Every text is cleaned to what a reader sees and its personal data redacted
(builder/privacy.py); rows are dropped that hold nothing to classify (no letter, digit or emoji once mentions, links
and placeholders are removed), only a removal notice, or a bot's message. Duplicates are merged: rows a model would
see as the same input once case, mentions, links, placeholders, punctuation and a retweet's "RT" are removed
(identical texts, for those with no words) become one row, with its raw text where it has one, the hate level all its sources allow (unknown and flagged as a
conflict when they disagree), the union of their targets, a soft label pooled over their annotators, and each source's
own label, id and author as provenance. Near-duplicates stay separate rows that share a group, so a split by group is
free of leakage. No language filter: identifiers misread slang, hashtags and African American English as other
languages (4 of 40 sampled non-English calls were right), so a filter would remove more English than not.

Run from the repository root: uv run --isolated --only-group builder python -m builder.build"""
import dataclasses
import hashlib
import html
import importlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

from builder.card import TABLE, markdown
from builder.card import write as write_card
from builder.privacy import redact
from builder.schema import EXPLICITNESS, HATE_LEVELS, SOFT_LEVELS, TARGET_CATEGORIES, TARGET_GROUPS, TEXT_FORMS

SOURCES = ("hatexplain", "ghc", "mhs", "cad", "sbf", "eap", "mlma", "ethos", "implicit_hate", "jigsaw",
           "davidson", "slur", "imsypp", "wsf", "intervene", "foxnews", "crehate", "olid", "hateval", "cmsb", "hasoc")
OPEN = ("CC0-1.0", "MIT", "CC-BY-4.0")
TIERS = {  # each table's own licence and the source licences that allow it; share-alike and non-commercial cannot mix
    "open": ("CC-BY-4.0", OPEN),
    "share_alike": ("CC-BY-SA-4.0", (*OPEN, "CC-BY-SA-4.0", "CC-BY-SA-3.0-ES")),
    "non_commercial": ("CC-BY-NC-SA-4.0", (*OPEN, "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0")),
    "all": (None, None),
}
OUT = Path("data/build")
ROW_GROUP = 50_000  # rows per Parquet row group, far below the 286 MiB a row group may hold for the Hub's viewer
REMOVED = ("[removed]", "[deleted]")
AUTOMATED = re.compile(r"(?i)\bI(?: am|'m) a bot\b")  # a bot's own signature; humans mocking one ("I am not a bot") stay
NOT_CONTENT = re.compile(r"https?://\S+|\bwww\.\S+|@user|@URL|u/user|<\w+>")
CLITIC = re.compile(r"\s+(?=n?['’](?:s|re|m|ll|ve|d|t)\b)")  # a tokeniser's space before a clitic: "it 's", "do n't"
NEAR = 0.8
INVISIBLE = re.compile(  # not U+200C, U+200D or tag characters: they bind emoji, flags and some scripts
    "[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f\\u00ad\\u200b\\u200e\\u200f\\u202a-\\u202e\\u2060-\\u2064\\u2066-\\u2069\\ufeff]")
CONTINUATION = "".join(bytes([b]).decode("cp1252", "ignore") for b in range(0x80, 0xc0))
MOJIBAKE = re.compile(f"[\\u00c2-\\u00f4][{re.escape(CONTINUATION)}]+")  # a UTF-8 sequence read as Windows-1252


def unmojibake(match: re.Match) -> str:
    try:
        return match.group().encode("cp1252").decode("utf-8")
    except UnicodeError:
        return match.group()


def restore(text: str) -> str:
    """The text with Windows-1252 characters that were decoded as control codes put back."""
    return re.sub("[\\x80-\\x9f]", lambda m: m.group().encode("latin-1").decode("cp1252", "ignore"), text)


def settle(step, text: str) -> str:
    """The text with step applied until it stops changing; each change shortens the text or removes a control code
    for good, so this ends."""
    while (after := step(text)) != text:
        text = after
    return text


def clean(text: str) -> str:
    """The text as a reader sees it: HTML entities decoded however deeply they were escaped; Windows-1252 characters
    decoded as control codes restored and UTF-8 read as Windows-1252 repaired wherever it occurs, together, since each
    can expose the other, until neither finds more; control codes and invisible formatting removed (soft hyphens among
    it, which also hide slurs from keyword filters); non-breaking spaces and line endings made plain; Unicode composed
    (NFC); ends trimmed."""
    text = settle(html.unescape, text)
    text = settle(lambda t: MOJIBAKE.sub(unmojibake, restore(t)), text)
    text = INVISIBLE.sub("", text).replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def should_keep(text: str) -> bool:
    """Whether a cleaned text stays: something is left to classify once mentions, links and placeholders are removed
    (a letter, digit or emoji), and it is neither a removal notice nor a bot's message."""
    rest = NOT_CONTENT.sub("", text)
    return (text not in REMOVED and not AUTOMATED.search(text)
            and any(unicodedata.category(c)[0] in "LN" or unicodedata.category(c) == "So" for c in rest))


def key(text: str) -> str:
    """The text reduced to its words, so the same post matches across sources that tokenise ("it 's") or anonymise
    it differently, and a retweet ("RT @user: ...", however nested) matches the tweet; empty when nothing but
    mentions, links or placeholders remain."""
    text = re.sub(r"https?://\S+|\bwww\.\S+|@\w+|<\w+>|\[\w+\]", " ", CLITIC.sub("", text.lower()))
    text = re.sub(r"^(?:\W*rt\b)+", " ", text.replace("'", "").replace("’", ""))  # \b is Unicode-aware: "RTÉ" stays
    return " ".join(re.findall(r"[a-z0-9]+", text))


def inputs(cache: Path) -> list[tuple[str, int, str]]:
    """The files of a source's download, with their sizes and SHA-256 hashes, so a rebuild can show it began from the
    same inputs; papers and hidden download metadata are left out."""
    files = [cache] if cache.is_file() else sorted(
        p for p in cache.rglob("*") if p.is_file() and p.suffix != ".pdf"
        and not any(part.startswith(".") for part in p.relative_to(cache).parts))
    out = []
    for path in files:
        with path.open("rb") as f:
            out.append((str(path), path.stat().st_size, hashlib.file_digest(f, "sha256").hexdigest()))
    return out


def pool() -> tuple[list[dict], dict[str, dict], dict[str, tuple[str, str]]]:
    """Every available source's records as rows, with the counts and input files per source and the sources left out,
    with their licence and why."""
    rows, stats, missing = [], {}, {}
    for name in SOURCES:
        print(f"== {name}")
        module = importlib.import_module(f"builder.sources.{name}")
        try:
            records = module.build()
        except OSError as e:
            missing[name] = (module.LICENCE, f"{type(e).__name__}: {e}")
            continue
        if len({r.source_id for r in records}) != len(records):
            raise ValueError(f"{name}: source ids repeat, so rows could not be told apart")
        kept = [{"id": r.id, **dataclasses.asdict(r), "text": text} for r in records
                if should_keep(text := redact(clean(r.text)))]
        stats[name] = {"records": len(records), "dropped": len(records) - len(kept), "licence": records[0].licence,
                       "inputs": inputs(module.CACHE)}
        rows += kept
    return rows, stats, missing


def cluster(rows: list[dict]) -> dict[str, list[dict]]:
    """The rows grouped by key (by text, for those with no words), each group named by a hash of its key, so a text
    has the same id in every table and build."""
    groups = defaultdict(list)
    for row in rows:
        groups[key(row["text"]) or "\0" + row["text"]].append(row)
    return {hashlib.sha256(k.encode()).hexdigest()[:16]: members for k, members in groups.items()}


def families(clusters: dict[str, list[dict]]) -> dict[str, str]:
    """Each cluster's near-duplicate family: the smallest cluster id among those joined to it, directly or through
    others, by word sets overlapping by at least 0.8 (Jaccard, over texts of four or more words). Families are not
    merged, since near-duplicates are often different posts: a joke retold about another group differs by one word."""
    parent = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lsh, words = MinHashLSH(threshold=NEAR, num_perm=64), {}
    for cluster_id, members in clusters.items():
        if len(w := set(key(members[0]["text"]).split())) < 4:
            continue
        sketch = MinHash(num_perm=64)
        sketch.update_batch([x.encode() for x in w])
        for other in lsh.query(sketch):
            if len(w & words[other]) / len(w | words[other]) >= NEAR:
                a, b = find(cluster_id), find(other)
                parent[max(a, b)] = min(a, b)
        lsh.insert(cluster_id, sketch)
        words[cluster_id] = w
    return {cluster_id: find(cluster_id) for cluster_id in clusters}


def resolve(levels: set[str]) -> str:
    """The one hate level a cluster's sources allow, where not_hateful allows offensive or neither and unknown allows
    anything; unknown when they disagree."""
    narrow = levels - {"unknown", "not_hateful"}
    if len(narrow) > 1 or ("hateful" in narrow and "not_hateful" in levels):
        return "unknown"
    return narrow.pop() if narrow else "not_hateful" if "not_hateful" in levels else "unknown"


def merge(cluster_id: str, members: list[dict], family: str) -> dict:
    """One cluster as one row of the resource."""
    members.sort(key=lambda m: (m["text_form"] != "raw", m["id"]))
    levels = {m["hate"] for m in members}
    hate = resolve(levels)
    explicit = {m["explicitness"] for m in members} - {"unknown"}
    soft = [m for m in members if m["soft_label"]]
    counted = [m["n_annotators"] for m in members if m["n_annotators"]]
    groups = sorted({g for m in members for g in m["target_group"]})
    return {
        "id": cluster_id, "near_duplicate_group": family, "text": members[0]["text"],
        "text_form": members[0]["text_form"],
        "platform": next((m["platform"] for m in members if m["platform"] != "unknown"), "unknown"),
        "hate": hate, "conflict": hate == "unknown" and len(levels - {"unknown"}) > 1,
        "target_category": sorted({c for m in members for c in m["target_category"]}
                                  | {c for g in groups for c in TARGET_GROUPS[g]}),
        "target_group": groups,
        "explicitness": explicit.pop() if len(explicit) == 1 and hate not in ("neither", "not_hateful") else "unknown",
        "soft_label": {lv: sum(m["soft_label"][lv] * m["n_annotators"] for m in soft)
                       / sum(m["n_annotators"] for m in soft) for lv in SOFT_LEVELS} if soft else None,
        "n_annotators": sum(counted) if counted else None,
        "sources": sorted({m["source"] for m in members}), "licences": sorted({m["licence"] for m in members}),
        "provenance": [{k: m[k] for k in ("source", "source_id", "hate", "label_raw", "author_id")} for m in members],
    }


def verify(rows: list[dict], final: list[dict], allowed: tuple[str, ...] | None) -> None:
    """Stops the build unless the table holds: one row per id, text and matching key; every near-duplicate group named
    by one of its rows; every text one to keep and unchanged by cleaning and redacting it again; only sources whose
    licences the table allows; every value in its vocabulary; a conflict only where the level is unknown; explicitness
    only where the text may be hostile; soft labels summing to one; and every kept source row in exactly one row's
    provenance."""
    keys = [k for r in final if (k := key(r["text"]))]
    ids = {r["id"] for r in final}
    sourced = [(p["source"], p["source_id"]) for r in final for p in r["provenance"]]
    failures = {
        "duplicate id": len(final) - len(ids),
        "near-duplicate group not a row": sum(r["near_duplicate_group"] not in ids for r in final),
        "duplicate text": len(final) - len({r["text"] for r in final}),
        "duplicate key": len(keys) - len(set(keys)),
        "text to drop, unclean or unredacted": sum(not should_keep(r["text"]) or redact(clean(r["text"])) != r["text"]
                                                   for r in final),
        "licence the table does not allow": sum(allowed is not None and not set(r["licences"]) <= set(allowed)
                                                for r in final),
        "value outside its vocabulary": sum(
            r["hate"] not in HATE_LEVELS or r["text_form"] not in TEXT_FORMS or r["explicitness"] not in EXPLICITNESS
            or not set(r["target_category"]) <= set(TARGET_CATEGORIES) for r in final),
        "conflict without unknown": sum(r["conflict"] and r["hate"] != "unknown" for r in final),
        "explicitness without hostility": sum(r["explicitness"] != "unknown" and r["hate"] in ("neither", "not_hateful")
                                              for r in final),
        "soft label not summing to one": sum(abs(sum(r["soft_label"].values()) - 1) > 1e-9
                                             for r in final if r["soft_label"]),
        "source row lost or repeated": abs(len(sourced) - len(rows)) + len(sourced) - len(set(sourced)),
    }
    if failed := {name: n for name, n in failures.items() if n}:
        raise ValueError(f"the resource fails its quality checks: {failed}")


COUNTS = ("Rows", "Hateful", "Offensive", "Neither", "Not hateful", "Unknown", "Race or ethnicity target",
          "Hateful, race or ethnicity target")


def counts(rows: list[dict]) -> list[str]:
    levels = Counter(r["hate"] for r in rows)
    race = [r for r in rows if "race_ethnicity" in r["target_category"]]
    return [f"{n:,}" for n in (len(rows), levels["hateful"], levels["offensive"], levels["neither"],
                                levels["not_hateful"], levels["unknown"], len(race),
                                sum(r["hate"] == "hateful" for r in race))]


def report(rows: list[dict], final: list[dict], stats: dict[str, dict], missing: dict[str, tuple[str, str]],
           tier: str, licence: str | None) -> str:
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    lines = [f"# Build report: {tier}", "", f"Licensed {licence}: it holds only sources whose licences allow that."
             if licence else "For local research only: it holds sources whose licences cannot be combined, and some "
             "that state none.", "", "## Sources, as mapped", ""]
    lines += markdown(["Source", "Licence", "Dropped", *COUNTS],
                      [[n, s["licence"], f"{s['dropped']:,}", *counts(by_source[n])] for n, s in stats.items()])
    lines += ["", "## The resource", ""] + markdown(list(COUNTS), [counts(final)])
    family = Counter(r["near_duplicate_group"] for r in final)
    lines += ["", (f"{sum(len(r['provenance']) > 1 for r in final):,} rows merge duplicates "
                   f"({len(rows) - len(final):,} source rows absorbed); {sum(len(r['sources']) > 1 for r in final):,} "
                   f"rows come from more than one source; {sum(r['conflict'] for r in final):,} rows have sources "
                   f"that disagree on the hate level; {sum(n for n in family.values() if n > 1):,} rows fall in "
                   f"{sum(n > 1 for n in family.values()):,} near-duplicate groups of two or more. Every quality "
                   f"check passed."), ""]
    lines += [f"Not available: {name} ({why})." for name, (_, why) in missing.items()] or ["Every source was available."]
    lines += ["", "## Inputs", ""] + markdown(["Source", "File", "Bytes", "SHA-256"], [
        [name, path, f"{size:,}", digest] for name, s in stats.items() for path, size, digest in s["inputs"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows, stats, missing = pool()
    tables = {}
    for tier, (licence, allowed) in TIERS.items():
        kept = [r for r in rows if allowed is None or r["licence"] in allowed]
        clusters = cluster(kept)
        family = families(clusters)
        final = [merge(cluster_id, members, family[cluster_id]) for cluster_id, members in clusters.items()]
        verify(kept, final, allowed)
        tables[tier] = pa.Table.from_pylist(final), report(
            kept, final, {n: s for n, s in stats.items() if allowed is None or s["licence"] in allowed},
            {n: m for n, m in missing.items() if allowed is None or m[0] in allowed}, tier, licence)
    for tier, (table, text) in tables.items():
        (OUT / tier).mkdir(parents=True, exist_ok=True)
        pq.write_table(table, OUT / tier / TABLE, row_group_size=ROW_GROUP)
        (OUT / tier / "report.md").write_text(text, encoding="utf-8")
        print(text)
    write_card(OUT, {tier: licence for tier, (licence, _) in TIERS.items() if licence}, SOURCES)


if __name__ == "__main__":
    main()
