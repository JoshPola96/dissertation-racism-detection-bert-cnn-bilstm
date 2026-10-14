---
pretty_name: HateAtlas
license:
- cc-by-4.0
- cc-by-sa-4.0
- cc-by-nc-sa-4.0
language:
- en
task_categories:
- text-classification
task_ids:
- hate-speech-detection
tags:
- hate-speech
- racism
- offensive-language
- target-detection
- data-provenance
size_categories:
- 100K<n<1M
configs:
$yaml_configs
---

# HateAtlas

**A target-aware atlas of English hate-speech datasets, with each row's provenance kept.** HateAtlas maps
$source_count published datasets onto one schema: a hate level, the groups a text targets, and every source's own label,
id and author beside each text. The configurations published here hold $published_count of them, as their licences
allow, and the builder adds the rest. Its default configuration holds $default_rows unique texts, $default_race of them
aimed at a racial or ethnic group ($default_race_hateful hateful). A builder that anyone can rerun fetches every
source from its home, checks it against the numbers its authors published, cleans it, removes personal data and merges
duplicates.

> **Content warning.** The data contains hateful, racist and abusive language, quoted as posted apart from personal
> data.

## Why it exists

HateAtlas began as a racism-detection dissertation for the MSc in Big Data Analytics and Artificial Intelligence at
Letterkenny Institute of Technology (now part of the Atlantic Technological University), 2021. Its models, BERT-large
read by CNN and BiLSTM layers and fused with features of each tweet, learned racism from the annotations of Waseem and
Hovy (2016), and the dissertation concluded that new datasets were needed. A 2026
[rebuild](https://github.com/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm) showed why. The racist tweets
were written by nine users, one of whom wrote 96 to 99% of them, mostly about Islam (Arango et al., 2019; Wiegand et
al., 2019). Every competent model reached the same ceiling, and the models missed hand-written racist sentences aimed
at Asian and Bengali people, which the data did not cover; only one flagged the sentence aimed at Black people, just
over its threshold. They had learned the authors and their topic. Once no author appears in both training and test
data, hate F1 on that dataset falls to between 7 and 24 (Arango et al., 2019). Released as tweet IDs, much of it had
been deleted by 2021.

What racism research lacks is not one more dataset but the existing ones read together. Racism is annotated, under
different names and schemes, across many datasets, platforms and years. MetaHate (Piot et al., 2024) unified 36 of them
as binary hate labels, without targets, authors or provenance. HateAtlas keeps what that leaves out:

- **Targets.** Each text's targeted groups come from a controlled vocabulary, so racism is a filter rather than a
  separate dataset, and models can be tested group by group.
- **Breadth.** In the default configuration, $default_race_hateful hateful texts target $race_group_count racial or
  ethnic groups, drawn from $race_sources sources on $race_platforms platforms; the sources that name authors alone
  contribute $race_authors distinct authors of them.
- **Provenance.** Each source's own label and id stay on every row, so any source's definition can be applied and any
  source dropped.
- **Leakage control.** Exact duplicates are merged, near-duplicates grouped and authors kept as pseudonyms, so an
  evaluation can hold out texts, authors or whole sources.
- **Text that lasts.** Only sources that release their text are included, so nothing decays as posts are deleted.

## Configurations

$configs

Each configuration is built only from sources whose licences allow its own, since share-alike and non-commercial
material cannot be combined in one table. $builder_only can be added only by running the
[builder](https://github.com/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm/tree/main/builder), which also
builds a table of every source for local research.

The same tables, with their build reports and this card, are on [Kaggle](https://www.kaggle.com/datasets/joshpeter96/hateatlas)
and in the repository's [releases](https://github.com/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm/releases),
as `hateatlas-open.parquet`, `hateatlas-share_alike.parquet` and `hateatlas-non_commercial.parquet`.

## Quick start

```python
from datasets import load_dataset

atlas = load_dataset("JoshPola96/HateAtlas", "open", split="train").to_pandas()
racism = atlas[atlas.target_category.apply(lambda categories: "race_ethnicity" in categories)]
hateful_racism = racism[racism.hate == "hateful"]
```

To evaluate, split by `near_duplicate_group` (for example with scikit-learn's `GroupShuffleSplit`), so that
near-identical texts never fall on both sides; to test generalisation, hold out a source, or the authors that
`provenance` names. A word-count model trained on the other sources stays well above chance on each held-out source
but loses ground, most where a source's hate is specific (East Asian Prejudice, the Slur Corpus), so a score on mixed
sources overstates performance on new ones. A model's input is `text` alone: `provenance`, `soft_label`, `conflict`, `n_annotators` and the
target fields describe the labels, and `sources`, `licences`, `platform` and `text_form` identify sources whose share
of hate differs, so feeding any of them to a model leaks the answer.

## Racism in the default configuration

$race_groups

## Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | string | A hash of the text's normalised form, the same in every configuration and build |
| `near_duplicate_group` | string | Shared by texts whose word sets overlap by at least 0.8 (Jaccard), directly or through others; split by it to avoid leakage |
| `text` | string | The text as a reader sees it, with personal data replaced |
| `text_form` | string | `raw` as posted, or `preprocessed` where the source lower-cased, tokenised or replaced parts of it |
| `platform` | string | `twitter`, `reddit`, `gab`, `youtube`, `stormfront`, `fox_news`, `civil_comments` or `unknown` |
| `hate` | string | The hate level that all the text's sources allow (below) |
| `conflict` | bool | True where its sources disagree; `hate` is then `unknown` |
| `target_category` | list of strings | `race_ethnicity`, `religion`, `national_origin`, `gender`, `sexual_orientation`, `disability`, `age`, `political` or `other` |
| `target_group` | list of strings | Groups from a controlled vocabulary, such as `black`, `asian`, `jewish`, `latinx`, `muslim` and `women` |
| `explicitness` | string | `explicit`, `implicit` or `unknown`, where a source annotates it for hostile texts |
| `soft_label` | struct | The share of annotators choosing `hateful`, `offensive` and `neither`, pooled over the sources that release votes |
| `n_annotators` | int | The annotators behind the label |
| `sources` | list of strings | The sources holding the text |
| `licences` | list of strings | Their licences |
| `provenance` | list of structs | Per source: `source`, `source_id`, its mapped `hate`, its own label verbatim (`label_raw`) and a pseudonymous `author_id` where it names authors |

## Hate levels

| Level | Meaning |
|---|---|
| `hateful` | The source's annotators judged it hate: an attack on people for who they are |
| `offensive` | Abusive, insulting or vulgar, but not hate |
| `neither` | Neither hateful nor offensive |
| `not_hateful` | Not hate, where the source does not say whether it is offensive |
| `unknown` | The source's annotators did not settle it, or its sources disagree |

Sources define hate and offence differently (Fortuna et al., 2020), so each source's mapping is set out below and
`provenance` keeps its own label. Two differences matter most: some sources count casual profanity as offensive and
others as neither, and some count attacks on political opinion as hate (HASOC 2019) or any offensive post about a
group (Social Bias Frames).

## Sources

$sources

<details>
<summary>How each source was read and mapped</summary>

$notes

</details>

## How it was built

Each source is fetched from where its authors publish it, mapped to the schema and checked against the numbers in its
paper or card; a source that does not reproduce them stops the build. Texts are cleaned to what a reader sees (HTML
entities, mis-decoded characters and sources' own escaping undone, invisible characters removed, Unicode composed),
personal data is replaced, and rows are dropped that hold nothing to classify (no letter, digit or emoji once
mentions, links and placeholders are removed), only a removal notice, or a bot's message (signed "I am a bot";
people mocking bots are kept). Texts that match once case, mentions, links, placeholders, punctuation and a
retweet's "RT" are removed become one row, with the level that all its sources allow. Near-duplicates are not merged,
since many are different posts (a joke retold about another group differs by one word), but share a
`near_duplicate_group`. A quality gate checks every table before it is written, an independent audit recomputes every
value from the published table, and two builds from the same inputs are byte-identical.

## Known limitations

- **Definitions differ across sources** (see Hate levels). Where a definition matters, use each source's own label
  from `provenance`. Where two sources hold the same text, they agree on whether it is hateful as follows. Cohen's κ
  corrects agreement for chance: when one source almost never says hateful, κ falls towards zero however high the raw
  agreement, the kappa paradox, so both are given:

$agreement

- **Label reliability differs.** Where a source labels the same text twice, the two labels differ this often. A low
  rate is weaker evidence than a high one, since a source may have labelled a text once and copied the label to its
  repeats:

$reliability

- **Slurs are not hate, and models confuse them.** A word-count model trained on all sources contradicts mostly
  labels that call a text not hateful though it uses a slur: quoted, reclaimed, used in counter speech or merely
  offensive (the Slur Corpus, HateXplain, MLMA, Davidson et al.). These are not labelling errors but the hardest and
  most useful negatives, so do not "clean" them away; test models on them.
- **Identity words trip models.** The same model raised false alarms on non-hateful texts that mention Black, Jewish,
  white or gay people about three times as often as on other texts, since identity words co-occur with hate in
  training data (Dixon et al., 2018). Evaluate with subgroup metrics (Borkan et al., 2019), and balance training with
  the many non-hateful texts here that mention each group.
- **Sources leave signatures.** Sources differ in formatting (lower-casing, tokenisation, placeholders) and in how much
  hate they hold, so formatting can predict the label and a model can learn the source instead of the hate. In the
  default configuration, Cramér's V between being hateful and a text's sources is $v_sources, and its platform
  $v_platform (0 means no association, 1 that one determines the other). Topic marks a source most (EAP's COVID
  hashtags, HASOC's cricket, CMSB's cookery show), and so does how it wrote links (raw, `URL`, `@URL`, `<url>`) and
  numbers (`<number>` in HateXplain and SBF), though these say almost nothing about the label. Hateful share by
  platform there:

$platforms

- **Dialect bias.** Annotators have labelled African American English as offensive or hateful more often than other
  English (Sap et al., 2019; Davidson et al., 2019), and sources labelled that way carry the bias.
- **Some posts are cut short.** Platforms and sources truncated some tweets, which end in "…"; the full posts can no
  longer be recovered.
- **A missing target means "not annotated", not "no target":** several sources annotate targets only for hateful
  texts, and some not at all.
- **English, with a trace of other languages.** Of 40 texts drawn at random from those a language identifier (lingua)
  called non-English, 4 were (Spanish, Italian, Dutch, Tagalog); the rest were slang, hashtags and African American
  English, which identifiers misread as other languages (Blodgett et al., 2016). That puts other languages at about
  0.1% of texts. No language filter is applied: it would remove far more English than not, and African American
  English above all.

## Personal information

The texts are public posts released by their sources. HateAtlas replaces e-mail addresses (`<email>`), phone numbers
(`<phone>`), bank account numbers (`<account>`) and every mention of a user (`@user`, and `u/user` on Reddit). In the
default configuration the placeholders appear as $placeholders. Where a source names authors, they become keyed pseudonyms (HMAC-SHA-256): one
author keeps one id across sources, but the id cannot be reversed by hashing guessed names. Names written in a text,
usernames inside links, mentions run onto a word (`you@name`, which cannot be told from obfuscations such as `cr@p`
and are kept) and the posts themselves can still identify people. To have a text removed, e-mail
[josh19peter96@gmail.com](mailto:josh19peter96@gmail.com) with its `id`.

## Uses

HateAtlas is for research: detecting hate speech and racism, measuring bias across target groups and dialects, and
studying how definitions and annotation differ between datasets. It is not for moderating or profiling real people
without human review, nor for generating hateful text.

## Licences and attribution

Each configuration carries the licence shown above, and each row's `licences` field names its sources' licences; the
sources are credited under Sources. HateAtlas changes them: labels are mapped to one schema, texts are cleaned and
redacted, and duplicates are merged. Four sources are released under the MIT licence, whose notices are reproduced
here: Davidson et al., "Copyright (c) 2017 Tom Davidson"; MLMA, "Copyright (c) 2019 HKUST-KnowComp"; the Slur Corpus,
"Copyright (c) 2020 McGill Network Dynamics Lab"; and the Fox News comments of Gao and Huang (2017), whose repository's
licence reads "Copyright (c) 2010-2019 Google, Inc.". Each grants: "Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED."

## Citation

```bibtex
@misc{polaprayil2026hateatlas,
  author = {Polaprayil, Joshua Peter},
  title = {{HateAtlas}: A Target-Aware Atlas of English Hate-Speech Datasets},
  year = {2026},
  publisher = {Hugging Face},
  doi = {10.57967/hf/10899},
  howpublished = {\url{https://huggingface.co/datasets/JoshPola96/HateAtlas}}
}
```

Please cite the sources you use as well.

## References

- Arango, A., Pérez, J. and Poblete, B. (2019). [Hate Speech Detection is Not as Easy as You May Think: A Closer Look at Model Validation](https://doi.org/10.1145/3331184.3331262). SIGIR.
- Blodgett, S. L., Green, L. and O'Connor, B. (2016). [Demographic Dialectal Variation in Social Media: A Case Study of African-American English](https://aclanthology.org/D16-1120). EMNLP.
- Borkan, D., Dixon, L., Sorensen, J., Thain, N. and Vasserman, L. (2019). [Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification](https://doi.org/10.1145/3308560.3317593). WWW Companion.
- Davidson, T., Bhattacharya, D. and Weber, I. (2019). [Racial Bias in Hate Speech and Abusive Language Detection Datasets](https://aclanthology.org/W19-3504). Workshop on Abusive Language Online.
- Dixon, L., Li, J., Sorensen, J., Thain, N. and Vasserman, L. (2018). [Measuring and Mitigating Unintended Bias in Text Classification](https://doi.org/10.1145/3278721.3278729). AIES.
- Fortuna, P., Soler, J. and Wanner, L. (2020). [Toxic, Hateful, Offensive or Abusive? What Are We Really Classifying? An Empirical Analysis of Hate Speech Datasets](https://aclanthology.org/2020.lrec-1.838). LREC.
- Piot, P., Martín-Rodilla, P. and Parapar, J. (2024). [MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection](https://doi.org/10.1609/icwsm.v18i1.31445). ICWSM.
- Sap, M., Card, D., Gabriel, S., Choi, Y. and Smith, N. A. (2019). [The Risk of Racial Bias in Hate Speech Detection](https://aclanthology.org/P19-1163). ACL.
- Waseem, Z. and Hovy, D. (2016). [Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter](https://aclanthology.org/N16-2013). NAACL Student Research Workshop.
- Wiegand, M., Ruppenhofer, J. and Kleinbauer, T. (2019). [Detection of Abusive Language: the Problem of Biased Datasets](https://aclanthology.org/N19-1060). NAACL.

Joshua Peter Polaprayil · [josh19peter96@gmail.com](mailto:josh19peter96@gmail.com)
