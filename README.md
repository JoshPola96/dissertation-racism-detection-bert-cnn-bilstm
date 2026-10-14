<!-- README.md -->
# Racism detection: a dissertation rebuilt, and the HateAtlas dataset

> **Scope** · Dissertation for the MSc in Big Data Analytics and Artificial Intelligence, Letterkenny Institute of Technology (LYIT, now part of ATU), 2021. Published on GitHub on 2021-08-30, its code was not revisited until this audit and rebuild as a corrected, runnable tutorial in October 2026, when the repository also became the home of HateAtlas, a dataset that reads 21 published hate-speech datasets onto one schema, and of a second notebook, in progress. The notebook as submitted is kept, unchanged, in [`original/`](original/racism_detection_2021.ipynb).

> **Content warning.** The data contains racist and abusive language. The notebook masks slurs in everything it prints; HateAtlas quotes posts as written.

## In this repository

| Part | What it is | Where |
|---|---|---|
| Notebook 1 | The 2021 dissertation, corrected and rebuilt as a tutorial: racism detection with BERT-large, CNN and BiLSTM layers, and a hybrid with tweet features | [`racism_detection_bert_cnn_bilstm.ipynb`](racism_detection_bert_cnn_bilstm.ipynb), [below](#notebook-1-the-dissertation-rebuilt) |
| HateAtlas | 21 English hate-speech datasets read together on one schema (18 in the published tables), with targets and every source's own label kept | [Hugging Face](https://huggingface.co/datasets/JoshPola96/HateAtlas), [Kaggle](https://www.kaggle.com/datasets/joshpeter96/hateatlas), [`builder/`](builder/), [below](#hateatlas) |
| Notebook 2 | Current methods trained and tested on HateAtlas (in progress) | [below](#notebook-2) |

## Notebook 1: the dissertation, rebuilt

Can a model tell racist tweets from the rest? This notebook answers with the dissertation's design: BERT-large from TensorFlow Hub, CNN and BiLSTM layers on top, and a hybrid model that adds numeric features of each tweet. Corrected, the 2021 architecture's test PR-AUC rises from 0.30 to 0.79, and to 0.81 with two training settings tuned; the hybrid of hidden layers scores highest, 0.82, consistent with the dissertation's hypothesis. A paired statistical test then shows that the hybrids and a word-count baseline cannot be told apart from the corrected model, because the dataset sets a ceiling; and the notebook shows how the models fail on racism the dataset does not cover. Every 2021 step is still there, in order, presented as what 2021 did, what the rebuild found, and what the notebook does now. That makes it a tutorial and a gallery of working techniques for classifying tweets.

### What it teaches

| Section | Concept |
|---|---|
| 1 | Running TensorFlow Hub's BERT on current TensorFlow: Keras 2 compatibility, GPU memory under WSL, deterministic GPU training |
| 2 | Combining datasets: stacking versus joining; testing undocumented labels against the data; conflicting duplicates; combining annotators' labels |
| 3 | Two kinds of cleaning: light normalisation for BERT, aggressive cleaning for word counts; the 2021 tweet tools (tweet-preprocessor, ekphrasis, emoji, VADER, langdetect) |
| 4 | Splitting before choosing: one seeded, stratified train/validation/test split, and why a two-way split is not enough |
| 5 | Exploring the whole dataset: n-grams, word clouds, sentiment and style features, the countries tweets name |
| 6 | The accuracy paradox; precision, recall, F1 and PR-AUC; a TF-IDF baseline |
| 7 | BERT's pooled and sequence outputs; feature extraction; the 2021 CNN-BiLSTM head on both, measured; tuning by random search, confirmed with seeds, and the winner's curse |
| 8 | A numeric-feature model; which features carry the signal (mutual information, ablation); late and intermediate fusion, and a search over fusion designs |
| 9 | Testing once: bootstrap confidence intervals and a paired comparison; the confusion matrix; each model's decisions on new sentences, and shortcut learning |
| 10 | Saving the hybrid with its scaler and threshold, checking the reload, and classifying any text |
| 11 | Limits: author overlap, mixed sources, a narrow sample of racism, dialect bias, the limits of the 2021 stack |
| 12 | Pitfalls worth knowing in any text-classification project, with the check that catches each |

### Results

The test set is 3,383 tweets, 289 of them racist, scored once after every choice had been made on validation tweets. PR-AUC measures how well a model ranks racist tweets above the rest (guessing scores 0.085); F1 and recall are at each model's validation threshold. Training is deterministic, so a rerun on the same hardware and software reproduces these numbers. The searches and ablations behind the chosen settings ran as separate experiment scripts on the same split; notebook section 11 describes them.

| Model | Test PR-AUC (95% CI) | F1 | Recall |
|---|---|---|---|
| Always "not racist" | | 0 (accuracy 91.5%) | 0 |
| TF-IDF + logistic regression | 0.776 (0.728–0.818) | 0.717 | 0.727 |
| 2021 wiring: CNN-BiLSTM on BERT's `pooled_output` | 0.303 (0.259–0.355) | 0.382 | 0.436 |
| Corrected: CNN-BiLSTM on BERT's `sequence_output` | 0.788 (0.741–0.831) | 0.737 | 0.827 |
| Corrected and tuned (learning rate 2e-4, L2 1e-3) | 0.805 (0.754–0.849) | 0.766 | 0.820 |
| Numeric features only | 0.537 (0.479–0.602) | 0.536 | 0.599 |
| Hybrid, 2021 fusion of the two models' outputs | 0.792 (0.744–0.837) | 0.716 | 0.668 |
| Hybrid, fusion of hidden layers | 0.817 (0.769–0.861) | 0.777 | 0.824 |

- Feeding BERT's token vectors instead of its summary vector takes the 2021 architecture from 0.30 to 0.79.
- The tuned model, the untuned corrected model, both hybrids and the TF-IDF baseline cannot be told apart. In a paired bootstrap, every difference from the tuned model has a 95% interval that contains zero (hybrid of hidden layers: +0.012, from −0.013 to +0.035). With 289 racist test tweets a difference would need to be about 0.03 or more to be detected reliably, so this test set cannot settle the hybrid's small lead either way. The data sets the ceiling: its racist class is nine authors writing mostly about Islam.
- The corrected models learned that narrow version of racism. They miss hand-written racist sentences aimed at Asian and Bengali people, and only the tuned model flags the one aimed at Black people, just over its threshold: a case of shortcut learning (notebook section 9).
- The second Kaggle file's non-racist tweets are easy negatives: the text models and the baseline wrongly flag Waseem & Hovy's non-racist tweets 11 to 17 times as often as those.
- Fine-tuning BERT-large itself, measured during the rebuild on validation tweets, added about 0.01 PR-AUC at small learning rates and made the model worse at a standard one: with frozen features used this well, the data, more than the method, sets the limit (notebook section 11).

### How it compares

Most published results on this data are averaged over three classes (racism, sexism, neither); the comparable number is F1 for the racist class alone. This notebook's task is binary and its non-racist tweets include file 2's, but F1 for the racist class ignores correctly rejected tweets, which keeps it comparable, though not identical. The tuned model's test F1 of 0.77 (95% interval 0.73–0.80) matches or slightly exceeds the published results that hold up on Waseem & Hovy's tweets, 0.70 to 0.75 (Zhang and Luo, 2018; Arango et al., 2019, who corrected two earlier reports of 0.94 and 0.95 to 0.70 and 0.75), without fine-tuning BERT; on Waseem & Hovy's test tweets alone it is 0.77 as well, and the hybrid of hidden layers reaches 0.78. Fine-tuned BERT-base reaches about 0.86 on one random split (Mozafari et al., 2019, computed from their confusion matrix). Like those results, this one is measured on a random split, and Waseem & Hovy's racist tweets were written by nine users, one of whom wrote 96 to 99% of them (Arango et al., 2019; Wiegand et al., 2019), so all these scores overstate performance on racism from new authors: Arango et al. found the hateful class's F1 falling to between 7 and 24 once no user appeared in both training and test data.

The published studies downloaded Waseem & Hovy's tweets between 2016 and 2019, while most were still online. By 2021 many had been deleted, and with the merge the dissertation's dataset held about 1,000 racist tweets, half the published set. This rebuild recovers all 1,970 from the Kaggle copy, so it is compared on the same racist tweets as the published results.

### What the rebuild changed

| 2021 | What the rebuild found | Now |
|---|---|---|
| BERT's `pooled_output`, the output the model's documentation demonstrates, reshaped to `(1024, 1)` for the CNN and BiLSTM | the layers read the 1,024 numbers of one summary vector as if they were a sentence | `sequence_output`, one vector per token, with padding masked: test PR-AUC 0.30 → 0.79 |
| The two Kaggle files merged on their text, to keep tweets labelled by both | where they overlap, the labels agree 99.4% of the time, but they share only 6,723 of 23,260 tweets and 982 of 1,970 racist ones | stacked, dropping only texts the files label differently |
| A second split to line up the hybrid's three inputs | its test set included tweets the text model had trained on, which raised its score | one seeded split for every model |
| Accuracy, with a 0.5 cut-off | accuracy says little when most tweets are not racist, and class weights move the best cut-off | PR-AUC and F1; cut-off chosen on validation data |
| Quantile scaling fitted on all tweets | the standard practice fits it on training tweets only, so the test set stays unseen | fitted on training tweets only |
| Adam at 5e-5 and L2 0.1 for the new layers | 5e-5 is a rate for fine-tuning BERT itself, too slow for new layers, and L2 0.1 held the CNN back | 2e-4 and 1e-3, chosen by a search on validation tweets and confirmed with three seeds: test PR-AUC 0.79 → 0.81 |
| BERT run inside the model, up to 100 epochs, early stopping on training loss | BERT-large alone would take more than 10 hours for 100 epochs on a 4 GB laptop GPU | BERT run once per tweet (under 5 minutes); each model trains in minutes |

Each notebook section ends with the 2021 steps it covers and what changed; section 12 collects the checks worth reusing in any project.

### The dissertation

The dissertation was written in about a month in 2021. By then much of the Twitter data the topic needed had been deleted, so it combined the labelled datasets that remained, downloading tweet text and metadata with the Hydrator app. The design drew on the research literature, a YouTube tutorial on fusing models, and the TensorFlow Hub BERT tutorials of the time (section 7 traces the BERT code to them). A literature search during the rebuild found no work published before August 2021 that combines these parts to detect racism in tweets: BERT-large read by CNN and BiLSTM layers, fused with a second model of each tweet's sentiment and numeric features. Text had been fused with features of its authors for abuse detection before (Mishra et al., 2018; Founta et al., 2019), and BERT feeding CNN and BiLSTM layers was published for hate speech in 2022 (Khan et al., 2022). It used the BERT-large model and preprocessor packaged for TensorFlow Hub, the route TensorFlow's documentation gave Keras users then; the notebook keeps them, and section 11 sets out what that stack limits. Training ran on Google Colab and took hours; section 7 shows why. It was examined at viva, and the MSc was awarded.

The dissertation concluded that the hybrid model outperformed the text-only model, and that new datasets were needed for this research. The rebuild is consistent with the first conclusion but does not confirm it: with the data leak removed, the hybrid of hidden layers has the highest test PR-AUC and F1, 0.012 and 0.011 above the tuned text model, but the 95% interval of the PR-AUC difference (−0.013 to +0.035) includes zero. It supports the second: the racist class, nine authors writing mostly about Islam, sets a ceiling that every competent model reaches.

The hybrid also aimed to reduce bias. Its second branch, built on each tweet's sentiment, gives the model evidence beyond word choice, so the words alone carry less of each decision. Testing that needs tweets labelled by the groups a model might treat unfairly, such as the dialect-aligned tweets Davidson et al. (2019) used; this dataset has none, so the aim itself stays untested. The rebuild measured the mechanism behind it instead: trained with and without its sentiment features, the hybrid ranked slightly better with them in each of three seeds and raised fewer false alarms, most consistently on tweets of positive tone (notebook section 8). In current terms the hybrid is multimodal, fusing text with tabular features.

The Hydrator returned 35 fields with every tweet, among them the author's follower and following counts, profile name, description and location, account age and verification, and the tweet's likes, retweets and hashtags. These say who wrote a tweet and how others responded, which the text alone cannot, and published work has found them useful for abuse detection (Founta et al., 2019). The 2021 notebook kept the author fields: the hybrid's numeric branch took the follower and following counts with each tweet's sentiment, the locations were explored by country, and the profile text was set aside, since the dataset was small and imbalanced. Adding such metadata to BERT's input was considered and left out, as guidance on doing it was scarce at the time. None of it can be downloaded now (notebook section 2), so the rebuild's numeric branch uses 25 features computed from each tweet's text instead. On this dataset author features would also identify the nine authors of its racist tweets, so any gain from them would need a split by author to be trusted (Arango et al., 2019).

### Run it

The notebook needs Linux, or WSL2 on Windows: `tensorflow-text` has no Windows build. An NVIDIA GPU with 4 GB is enough; on the CPU, BERT-large is about 50 times slower. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm.git
cd dissertation-racism-detection-bert-cnn-bilstm
uv sync
```

Open `racism_detection_bert_cnn_bilstm.ipynb` in VS Code, select the `.venv` kernel and run all cells. The first run downloads BERT-large (about 1.3 GB) and the word statistics ekphrasis uses. Versions are pinned in `uv.lock`, and Python must be 3.12.1 to 3.13: on 3.12.0, a bug in Python's `zipfile` stops saved models from loading. Under WSL, keep the clone in the Linux file system (`~/`); a clone under `/mnt/c` works but installs and loads slowly.

## HateAtlas

The dissertation's conclusion that new datasets were needed led to [HateAtlas](https://huggingface.co/datasets/JoshPola96/HateAtlas): 21 English hate-speech datasets read together on one schema, with each text's hate level, the groups it targets, and every source's own label, id and author kept beside it. The published tables hold 18 of them, as their licences allow; the builder adds the other three on your machine. Racism becomes a filter across sources, platforms and authors instead of one dataset's nine authors. Its dataset card sets out the sources, how each was read, and the dataset's limits. Cite it by its DOI, [10.57967/hf/10899](https://doi.org/10.57967/hf/10899).

It is published in three configurations, one per licence tier, since share-alike and non-commercial sources cannot share a table: `open` (CC BY 4.0, the default), `share_alike` (CC BY-SA 4.0) and `non_commercial` (CC BY-NC-SA 4.0).

```python
from datasets import load_dataset

atlas = load_dataset("JoshPola96/HateAtlas", "open", split="train")
```

The same tables, their build reports and the dataset card can also be downloaded from this repository's [latest release](https://github.com/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm/releases/latest) and from [Kaggle](https://www.kaggle.com/datasets/joshpeter96/hateatlas).

To build it yourself, in Linux or WSL ([`builder/README.md`](builder/README.md) explains every step, the four gated sources, how the pipeline works and how to add a source):

1. Clone the repository: `git clone https://github.com/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm.git`, then `cd` into it.
2. Install [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`.
3. Set a private key for the author pseudonyms: `export AUTHOR_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"`.
4. Optionally, download the four gated sources (Jigsaw, Implicit Hate, HatEval, CMSB); the builder builds without any it cannot find.
5. Build: `uv run --isolated --only-group builder python -m builder.build` (about 20 minutes).
6. Find the tables, their reports and the dataset card in `data/build/`.

## Notebook 2

Among the published results found for this data, only fine-tuned BERT-base scores higher (see [How it compares](#how-it-compares)). A second notebook will take the study forward with current methods:

- PyTorch and Hugging Face in place of the TensorFlow Hub stack, with BERT fine-tuned inside the hybrid
- further pretraining on the task's own tweets before fine-tuning (Gururangan et al., 2020)
- user metadata as BERT's second text segment, where a dataset still provides it
- newer encoders and large language models as baselines
- tests beyond a random split: functional tests (Röttger et al., 2021) and bias measured across groups
- current racism datasets, so the racist class is not the work of nine authors: now [HateAtlas](#hateatlas), above

## Repository contents

| Path | What it is |
|---|---|
| `racism_detection_bert_cnn_bilstm.ipynb` | The tutorial, with outputs; if GitHub's preview is slow to load, [view it on nbviewer](https://nbviewer.org/github/JoshPola96/dissertation-racism-detection-bert-cnn-bilstm/blob/main/racism_detection_bert_cnn_bilstm.ipynb) |
| `original/racism_detection_2021.ipynb` | The notebook as submitted in 2021, unchanged (identical to commit `e185b37`). Published in August 2021 as a code gallery for practitioners and students, it ran on Google Colab with data files prepared outside it, so it records the original steps rather than running as it stands; the rebuilt notebook is the runnable version |
| `data/` | The source files (below) |
| `models/hybrid.keras`, `models/feature_scaler.pkl`, `models/hybrid.json` | The hybrid's trained layers (BERT-large reloads from TensorFlow Hub), its fitted feature scaler and its decision threshold; section 10 loads them and classifies any text |
| `builder/` | The HateAtlas builder: one module per source in `builder/sources/`, the shared schema, the privacy rules and the dataset card |
| `pyproject.toml`, `uv.lock` | Dependencies, pinned; the builder's are in the `builder` group |
| `CITATION.cff` | How to cite this repository and HateAtlas |
| `LICENSE` | MIT licence for the code |

## Data and licence

| File | Source |
|---|---|
| `twitter_racism_parsed_dataset.csv` | [Kaggle: racism-tweets](https://www.kaggle.com/datasets/raghadabdullah/racism-tweets) |
| `classified_tweets.csv` | [Kaggle: classified-tweets](https://www.kaggle.com/datasets/munkialbright/classified-tweets) |
| `NAACL_SRW_2016.csv`, `NLP+CSS_2016.csv` | [Waseem & Hovy's annotations](https://github.com/zeeraktalat/hatespeech): tweet IDs and labels |

The racist tweets in both Kaggle files are those annotated by Waseem and Hovy (2016); about half the second file's non-racist tweets come from other, undocumented collections (notebook section 2). Each dataset remains under its source's terms. HateAtlas's tables carry their configuration's licence (CC BY 4.0, CC BY-SA 4.0 or CC BY-NC-SA 4.0) and credit every source in the dataset card. The code, the builder included, is under the MIT licence; [`CITATION.cff`](CITATION.cff) says how to cite it.

## References

- Alorainy, W., Burnap, P., Liu, H. and Williams, M. L. (2019). ["The Enemy Among Us": Detecting Cyber Hate Speech with Threats-based Othering Language Embeddings](https://doi.org/10.1145/3324997). ACM Transactions on the Web.
- Arango, A., Pérez, J. and Poblete, B. (2019). [Hate Speech Detection is Not as Easy as You May Think: A Closer Look at Model Validation](https://doi.org/10.1145/3331184.3331262). SIGIR.
- Bergstra, J. and Bengio, Y. (2012). [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html). Journal of Machine Learning Research.
- Davidson, T., Bhattacharya, D. and Weber, I. (2019). [Racial Bias in Hate Speech and Abusive Language Detection Datasets](https://aclanthology.org/W19-3504/). Workshop on Abusive Language Online.
- Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017). [Automated Hate Speech Detection and the Problem of Offensive Language](https://arxiv.org/abs/1703.04009). ICWSM.
- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/). NAACL.
- Founta, A.-M., Chatzakou, D., Kourtellis, N., Blackburn, J., Vakali, A. and Leontiadis, I. (2019). [A Unified Deep Learning Architecture for Abuse Detection](https://doi.org/10.1145/3292522.3326028). WebSci.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M. and Wichmann, F. A. (2020). [Shortcut learning in deep neural networks](https://doi.org/10.1038/s42256-020-00257-z). Nature Machine Intelligence.
- Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D. and Smith, N. A. (2020). [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/). ACL.
- Khan, S., Fazil, M., Sejwal, V. K., Alshara, M. A., Alotaibi, R. M., Kamal, A. and Baig, A. R. (2022). [BiCHAT: BiLSTM with deep CNN and hierarchical attention for hate speech detection](https://doi.org/10.1016/j.jksuci.2022.05.006). Journal of King Saud University – Computer and Information Sciences.
- Mishra, P., Del Tredici, M., Yannakoudakis, H. and Shutova, E. (2018). [Author Profiling for Abuse Detection](https://aclanthology.org/C18-1093/). COLING.
- Mozafari, M., Farahbakhsh, R. and Crespi, N. (2019). [A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media](https://arxiv.org/abs/1910.12574). Complex Networks.
- Peters, M. E., Ruder, S. and Smith, N. A. (2019). [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://aclanthology.org/W19-4302/). RepL4NLP workshop.
- Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H. and Pierrehumbert, J. (2021). [HateCheck: Functional Tests for Hate Speech Detection Models](https://aclanthology.org/2021.acl-long.4/). ACL.
- Waseem, Z. (2016). [Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter](https://aclanthology.org/W16-5618/). Workshop on NLP and Computational Social Science.
- Waseem, Z. and Hovy, D. (2016). [Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter](https://aclanthology.org/N16-2013/). NAACL Student Research Workshop.
- Wiegand, M., Ruppenhofer, J. and Kleinbauer, T. (2019). [Detection of Abusive Language: the Problem of Biased Datasets](https://aclanthology.org/N19-1060/). NAACL.
- Zhang, Z. and Luo, L. (2018). [Hate Speech Detection: A Solved Problem? The Challenging Case of Long Tail on Twitter](https://arxiv.org/abs/1803.03662). Semantic Web.

Joshua Peter Polaprayil · [josh19peter96@gmail.com](mailto:josh19peter96@gmail.com)
