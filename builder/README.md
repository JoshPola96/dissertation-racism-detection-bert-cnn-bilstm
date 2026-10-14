<!-- builder/README.md -->
# The HateAtlas builder

This package builds [HateAtlas](https://huggingface.co/datasets/JoshPola96/HateAtlas) from its 21 sources. It fetches each source from where its authors publish it, checks it against the numbers they published, maps it to one schema, cleans it, removes personal data and merges duplicates. It then writes one Parquet table per licence tier, a report beside each, and the dataset card. The finished tables are on Hugging Face. Build them yourself to verify them, to add the three sources whose licences keep them off the Hub (OLID, HASOC 2019, ETHOS), or to extend HateAtlas.

## Build it

Tested on Linux (Ubuntu under WSL 2). A build needs about 1.5 GB of disk and takes about 20 minutes.

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm.git
   cd dissertation-racism-detection-bert-cnn-bilstm
   ```

2. Install [uv](https://docs.astral.sh/uv/), which fetches a suitable Python and the builder's two dependencies, datasketch and pyarrow:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Choose a secret key for the author pseudonyms, and keep it private. The key is what stops anyone from reversing a pseudonym by hashing guessed usernames. Your author ids will differ from the published ones; everything else will be identical. To get the same ids in every build, save the key and export it again each time.

   ```bash
   export AUTHOR_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
   ```

4. Optionally, download the four gated sources (below). Without them, the builder reports them as missing and builds the rest.

5. Run the build:

   ```bash
   uv run --isolated --only-group builder python -m builder.build
   ```

6. Find the outputs in `data/build/`:
   - `README.md`, the dataset card;
   - one folder per tier, each holding `hateatlas.parquet` and `report.md`:
     - `open/`, `share_alike/` and `non_commercial/`, as published;
     - `all/`, every source, for local research only, since its licences cannot be combined.

   Each report lists every input file's size and SHA-256, so two builds can be shown to have started from the same downloads. Builds from the same inputs and the same key are byte-identical.

### The four gated sources

Their authors ask each user to accept their terms, so they cannot be fetched for you:

| Source | Get access | Put it at |
|---|---|---|
| Jigsaw Unintended Bias | Accept the [competition's rules](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/rules), set up the [Kaggle API](https://www.kaggle.com/docs/api), then run `kaggle competitions download jigsaw-unintended-bias-in-toxicity-classification -f all_data.csv -p data/sources/jigsaw` | `data/sources/jigsaw/all_data.csv.zip` |
| Implicit Hate | Complete the [authors' survey](https://forms.gle/QxCpEbVp91Z35hWFA) and save the zip it leads to | `data/sources/implicit_hate/implicit-hate-corpus-nov-2021.zip` |
| HatEval | Accept the terms of [valeriobasile/HatEval](https://huggingface.co/datasets/valeriobasile/HatEval), log in with `uvx --from huggingface_hub hf auth login`, then run `uvx --from huggingface_hub hf download valeriobasile/HatEval --repo-type dataset --local-dir data/sources/hateval` | `data/sources/hateval/` |
| "Call me sexist, but" | Register with GESIS and download the [dataset](https://doi.org/10.7802/2251) | `data/sources/cmsb/sexism_data.csv` and `data/sources/cmsb/sexism_annotations.csv` |

## How it works

```
for each source in sources/   fetch → map to the schema → check against its paper's numbers
pool every row                clean the text → redact personal data → drop rows with nothing to
                              classify, removal notices and bot messages
for each licence tier         merge exact duplicates (same normalised text) into one row,
                              group near-duplicates → quality gate → table and report
finally                       the dataset card, filled from the tables and the source modules
```

The build stops when something is wrong: a source that does not reproduce its paper, a record outside the schema's vocabularies, or a table that fails the quality gate. Nothing is written until every tier passes.

| File | Role |
|---|---|
| `build.py` | The pipeline: pooling, cleaning, the duplicate key, merging, near-duplicate groups, licence tiers, the quality gate and the reports |
| `schema.py` | The record every source maps into, and the controlled vocabularies: hate levels, target categories and target groups |
| `privacy.py` | Redaction of e-mail addresses, phone and account numbers and mentions, and the keyed author pseudonyms |
| `card.md`, `card.py` | The dataset card's template, and the code that fills it with numbers read from the tables |
| `sources/<name>.py` | One module per source |

### The quality gate

Every table must pass these checks before it is written:

- one row per id, per text and per normalised key;
- every near-duplicate group named by one of its rows;
- every text one to keep, and unchanged by cleaning and redacting it again;
- only licences the table allows;
- every value in its vocabulary;
- a conflict only where the level is unknown;
- explicitness only where a text may be hostile;
- soft labels summing to one;
- every kept source row in exactly one row's provenance.

## Adding a source

A source module defines:

- a docstring whose first paragraph opens with the source's name and says how the source is read and mapped (the card shows it);
- `URL`, `CACHE` (where its download lives), `LICENCE` (an SPDX id) and `CITATION`;
- `fetch()`, which downloads once into `CACHE` and returns the rows, raising `FileNotFoundError` with instructions where the download is gated;
- `records(rows)`, which maps the rows to `schema.Record`s;
- `check(rows, records)`, which fails unless the rows reproduce numbers the authors published;
- `build()`, which runs the three.

Add its name to `SOURCES` in `build.py`. If its licence allows a published tier, add the licence to that tier in `TIERS`. Check the module on its own first:

```bash
uv run --isolated --only-group builder python -m builder.sources.<name>
```
