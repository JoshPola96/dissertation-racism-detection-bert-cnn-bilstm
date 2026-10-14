# builder/card.py
"""Writes the dataset card that Hugging Face shows for the published tables (card.md, filled in). Every number is read
from the tables, and every source's name, account, licence and citation from its module, so the card cannot drift from
what it describes."""
import importlib
import inspect
import math
import re
import string
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq

from builder.schema import TARGET_GROUPS

TABLE = "hateatlas.parquet"
TEMPLATE = Path(__file__).with_name("card.md")
RACE = tuple(group for group, categories in TARGET_GROUPS.items() if "race_ethnicity" in categories)
PLACEHOLDERS = ("@user", "u/user", "<email>", "<phone>", "<account>")


def markdown(header: list[str], body: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(row) + " |" for row in [header, ["---"] * len(header), *body]]


def summary(path: Path) -> dict:
    """The counts the card reports for one table."""
    rows = pq.read_table(path, columns=["text", "platform", "hate", "target_category", "target_group",
                                       "provenance"]).to_pylist()
    texts, hateful, repeated, differing = Counter(), Counter(), Counter(), Counter()
    group_texts, group_hateful, group_sources = Counter(), Counter(), defaultdict(set)
    race_sources, race_platforms, race_authors = set(), set(), set()
    votes, by_sources, by_platform = defaultdict(Counter), Counter(), Counter()
    for r in rows:
        own = defaultdict(list)
        for p in r["provenance"]:
            texts[p["source"]] += 1
            hateful[p["source"]] += p["hate"] == "hateful"
            own[p["source"]].append(p["hate"])
        for source, levels in own.items():
            definite = set(levels) & {"hateful", "offensive", "neither"}
            if len(levels) > 1 and definite:
                repeated[source] += 1
                differing[source] += len(definite) > 1
        said = {s: {lv == "hateful" for lv in levels if lv != "unknown"} for s, levels in own.items()}
        for a, b in combinations(sorted(s for s, v in said.items() if len(v) == 1), 2):
            votes[a, b][next(iter(said[a])), next(iter(said[b]))] += 1
        by_sources[tuple(sorted(own)), r["hate"] == "hateful"] += 1
        by_platform[r["platform"], r["hate"] == "hateful"] += 1
        for group in set(r["target_group"]) & set(RACE):
            group_texts[group] += 1
            if r["hate"] == "hateful":
                group_hateful[group] += 1
                group_sources[group].update(p["source"] for p in r["provenance"])
        if r["hate"] == "hateful" and "race_ethnicity" in r["target_category"]:
            race_sources.update(p["source"] for p in r["provenance"])
            race_platforms.add(r["platform"])
            race_authors.update(p["author_id"] for p in r["provenance"] if p["author_id"])
    return {
        "rows": len(rows), "levels": Counter(r["hate"] for r in rows),
        "race": sum("race_ethnicity" in r["target_category"] for r in rows),
        "race_hateful": sum(r["hate"] == "hateful" and "race_ethnicity" in r["target_category"] for r in rows),
        "texts": texts, "hateful": hateful, "repeated": repeated, "differing": differing,
        "group_texts": group_texts, "group_hateful": group_hateful, "group_sources": group_sources,
        "race_sources": race_sources, "race_platforms": race_platforms - {"unknown"}, "race_authors": race_authors,
        "platforms": Counter(r["platform"] for r in rows),
        "platform_hateful": Counter(r["platform"] for r in rows if r["hate"] == "hateful"),
        "placeholders": {p: sum(p in r["text"] for r in rows) for p in PLACEHOLDERS},
        "votes": votes, "v_sources": cramers_v(by_sources), "v_platform": cramers_v(by_platform),
    }


def kappa(votes: Counter) -> tuple[int, float, float, float, float]:
    """Shared texts, raw agreement, Cohen's kappa and each source's hateful share, from two sources' (hateful,
    hateful) votes; kappa is nan when chance alone predicts every vote."""
    n = sum(votes.values())
    agree = (votes[False, False] + votes[True, True]) / n
    a, b = (votes[True, False] + votes[True, True]) / n, (votes[False, True] + votes[True, True]) / n
    chance = a * b + (1 - a) * (1 - b)
    return n, agree, (agree - chance) / (1 - chance) if chance < 1 else math.nan, a, b


def cramers_v(counts: Counter) -> float:
    """Cramér's V between a category and being hateful, from counts of (category, hateful) pairs."""
    n = sum(counts.values())
    rows, cols = Counter(), Counter()
    for (x, y), k in counts.items():
        rows[x] += k
        cols[y] += k
    chi2 = sum((counts[x, y] - rows[x] * cols[y] / n) ** 2 / (rows[x] * cols[y] / n) for x in rows for y in cols)
    return math.sqrt(chi2 / n)  # with a yes-or-no label, min(categories, 2) - 1 is 1


def note(module) -> tuple[str, str]:
    """A source's name and the first paragraph of its docstring, which says how it is read and mapped."""
    paragraph = " ".join(inspect.getdoc(module).split("\n\n")[0].split())
    return paragraph.split(" (", 1)[0], paragraph


def code(text: str) -> str:
    """The text with placeholders such as <user> as code, which a Markdown renderer would otherwise drop as HTML."""
    return re.sub(r"<\w+>", lambda m: f"`{m.group()}`", text)


def listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def write(out: Path, tiers: dict[str, str], sources: tuple[str, ...]) -> None:
    """Writes out/README.md for the tables in out/<tier>/, the first tier being the default configuration."""
    stats = {tier: summary(out / tier / TABLE) for tier in tiers}
    modules = {name: importlib.import_module(f"builder.sources.{name}") for name in sources}
    names = {name: note(module)[0] for name, module in modules.items()}
    home = {}
    for tier in tiers:
        for source in stats[tier]["texts"]:
            home.setdefault(source, tier)
    default = stats[next(iter(tiers))]
    n = "{:,}".format

    configs = markdown(
        ["Configuration", "Licence", "Sources", "Texts", "Hateful", "Offensive", "Neither", "Not hateful", "Unknown",
         "Race or ethnicity target", "Hateful, race or ethnicity target"],
        [[f"`{tier}`", licence, str(len(s["texts"])), n(s["rows"]),
          *(n(s["levels"][level]) for level in ("hateful", "offensive", "neither", "not_hateful", "unknown")),
          n(s["race"]), n(s["race_hateful"])] for (tier, licence), s in zip(tiers.items(), stats.values())])
    race_groups = markdown(
        ["Group", "Texts", "Hateful", "Sources of the hateful texts"],
        [[f"`{g}`", n(default["group_texts"][g]), n(default["group_hateful"][g]),
          listed(sorted(names[s] for s in default["group_sources"][g])) if default["group_sources"][g] else "none"]
         for g in sorted(default["group_texts"], key=lambda g: (-default["group_hateful"][g], g))])
    table = []
    for name in sources:
        url = re.search(r"https?://[^\s;]+", modules[name].CITATION).group().rstrip(".")
        s = stats[home[name]] if name in home else None
        table.append([f"[{names[name]}]({url})", modules[name].LICENCE,
                      ", ".join(f"`{t}`" for t in tiers if name in stats[t]["texts"]) or "builder only",
                      n(s["texts"][name]) if s else "", n(s["hateful"][name]) if s else ""])
    notes = [f"**{names[name]}** · {modules[name].LICENCE}\n\n{code(note(modules[name])[1])}\n\n*Cite:* "
             f"{modules[name].CITATION}" for name in sources]
    reliability = markdown(
        ["Source", "Texts labelled twice", "Labelled differently"],
        [[names[name], n(stats[home[name]]["repeated"][name]),
          f"{stats[home[name]]['differing'][name] / stats[home[name]]['repeated'][name]:.1%}"]
         for name in sorted((s for s in home if stats[home[s]]["repeated"][s]),
                            key=lambda s: -stats[home[s]]["repeated"][s])])
    platforms = markdown(
        ["Platform", "Texts", "Hateful"],
        [[f"`{p}`", n(count), f"{default['platform_hateful'][p] / count:.1%}"]
         for p, count in default["platforms"].most_common()])
    pairs = {}
    for tier in tiers:
        for pair, votes in stats[tier]["votes"].items():
            pairs.setdefault(pair, votes)
    agreement = markdown(
        ["Sources", "Shared texts", "Agree on hateful or not", "Cohen's κ", "Hateful share, each"],
        [[f"{names[a]} · {names[b]}", n(k), f"{agree:.1%}", "undefined" if math.isnan(kap) else f"{kap:.2f}",
          f"{sa:.0%} · {sb:.0%}"]
         for (a, b), (k, agree, kap, sa, sb) in sorted(((p, kappa(v)) for p, v in pairs.items()),
                                                         key=lambda x: -x[1][0]) if k >= 100])
    counts = [f"`{p}` in {n(k)} texts" if k else f"`{p}` in none" for p, k in default["placeholders"].items()]
    yaml = [f"- config_name: {tier}\n  data_files: {tier}/{TABLE}" for tier in tiers]
    yaml[0] += "\n  default: true"
    fields = {
        "yaml_configs": "\n".join(yaml),
        "source_count": str(len(sources)), "default_rows": n(default["rows"]), "default_race": n(default["race"]),
        "default_race_hateful": n(default["race_hateful"]),
        "race_group_count": str(sum(1 for g in default["group_hateful"] if default["group_hateful"][g])),
        "race_sources": str(len(default["race_sources"])), "race_platforms": str(len(default["race_platforms"])),
        "race_authors": n(len(default["race_authors"])), "configs": "\n".join(configs),
        "builder_only": listed([f"{names[s]} ({modules[s].LICENCE})" for s in sources if s not in home]),
        "race_groups": "\n".join(race_groups),
        "sources": "\n".join(markdown(["Source", "Licence", "Configurations", "Texts", "Hateful"], table)),
        "published_count": str(len(home)),
        "notes": "\n\n".join(notes), "reliability": "\n".join(reliability), "platforms": "\n".join(platforms),
        "placeholders": listed(counts), "agreement": "\n".join(agreement),
        "v_sources": f"{default['v_sources']:.2f}", "v_platform": f"{default['v_platform']:.2f}",
    }
    card = string.Template(TEMPLATE.read_text(encoding="utf-8")).substitute(fields)
    (out / "README.md").write_text(card, encoding="utf-8")
