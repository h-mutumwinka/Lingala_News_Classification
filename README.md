# Lingala News Topic Classification

Automatic topic classification of **Lingala** news articles into **politics, health, sports and business**, using the
Lingala part of [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews) (Adelani et al., 2023).

ALU · NLP and Language Technologies · Summative project (Option 3: Text Classification for an African Language)
Author: Joseph Nishimwe

| Resource | Link |
|---|---|
| Live web app | `https://huggingface.co/spaces/YOUR-HF-USERNAME/lingala-news-topic-classifier` *(add after deployment)* |
| Fine-tuned model | `https://huggingface.co/YOUR-HF-USERNAME/lingala-news-topic-classifier` *(add after notebook 06)* |
| Demo video | *(add link)* |
| Report | *(add link)* |

## 1. Problem

Lingala is spoken by about 40 million people (roughly 20 million first-language and 20 million second-language speakers) in the
Democratic Republic of the Congo and the Republic of the Congo (Eberhard et al., 2024). Kinshasa and Brazzaville media publish in it, but
Lingala has very few NLP tools: commercial text-classification APIs do not support it (Google Cloud Natural Language content
classification covers 12 languages, none African), and it is absent from most pretrained language models.

A topic classifier lets news aggregators, archives, media monitors and NGOs route and search Lingala content automatically.
The task: given a Lingala article (headline + text), predict one of four topics.

## 2. Dataset

| | |
|---|---|
| Source | MasakhaNEWS, Lingala (`lin`) configuration; articles from [VOA Lingala](https://www.voalingala.com) |
| Size | 870 articles: 608 train / 87 validation / 175 test (official splits, kept unchanged) |
| Labels | politics 500 (57%), health 193 (22%), sports 95 (11%), business 82 (9%) |
| Length | median 165 words (headline + text), max 641 |
| Annotation | 2 annotators, Fleiss' kappa 0.56 (Adelani et al., 2023) |
| Licence | See the dataset card on Hugging Face |

`src/data_preparation.py` downloads the data (Hugging Face, with the Masakhane GitHub repository as fallback), then
normalises Unicode (NFC), replaces typographic quotes, removes URLs, stray symbols and extra whitespace, builds the model input
`content = headline + ". " + text`, and checks that no article appears in two splits. We keep the accents (they carry tone and
French loanwords) and test removing them as an experiment.

Key findings from the EDA (`notebooks/01_eda.ipynb`): strong class imbalance; 53% of the training vocabulary occurs only once and about
7% of test tokens are unseen in training; frequent French code-mixing (*ba stocks*, *procès*, *Léopards*); inconsistent accents and
spellings (*président/president*, *Tshisekedi/Tshiskedi*); articles concentrated in 2018–2023 (Ebola, COVID-19).

## 3. Methodology

| Notebook | What it does | Runs on |
|---|---|---|
| `01_eda.ipynb` | Exploratory data analysis | CPU |
| `02_bag_of_words_baselines.ipynb` | Majority class, TF-IDF + Naive Bayes / Logistic Regression / linear SVM; feature, input and preprocessing experiments; 5-fold × 3 cross-validation | CPU, ~10 min |
| `03_word2vec.ipynb` | Word2Vec trained on the training articles; averaged document vectors + logistic regression | CPU, ~3 min |
| `04_rnn.ipynb` | BiLSTM / BiGRU with attention, random vs Word2Vec embeddings, frozen vs fine-tuned, class weights | CPU, ~80 min |
| `05_transformer_from_scratch.ipynb` | Small BERT-style Transformer with its own WordPiece tokeniser, no pre-training (control) | CPU |
| `06_transformer_finetuning_colab.ipynb` | Fine-tuning XLM-R base vs AfroXLMR base (and 76L), truncation and class-weight ablations | **Colab GPU**, ~45-60 min |
| `07_results_and_error_analysis.ipynb` | Final comparison, bootstrap confidence intervals, significance tests, error analysis | CPU |

Evaluation protocol:
* Train on `train`, tune on `validation`, use `test` **once** per final model.
* Main metric **macro-F1** (all topics weigh the same). We also report weighted-F1 (the metric of the MasakhaNEWS paper), accuracy,
  macro precision/recall and per-class F1.
* Because the validation set is tiny (8 business articles), bag-of-words settings are chosen by 5-fold cross-validation repeated 3 times;
  neural models are retrained with 3 seeds (mean ± std); test scores come with 95% bootstrap intervals and paired bootstrap tests.

## 4. Results

Test set (175 articles), each final model evaluated once. Neural models: mean ± std over 3 seeds.

| Model | Macro-F1 | 95% bootstrap CI | Weighted-F1 | Accuracy | F1 business |
|---|---|---|---|---|---|
| Majority class | 0.182 | 0.165–0.196 | 0.416 | 0.571 | 0.000 |
| TF-IDF + Complement NB | 0.673 | 0.609–0.738 | 0.773 | 0.811 | 0.111 |
| TF-IDF (word + char) + linear SVM | 0.822 | 0.747–0.884 | 0.852 | 0.857 | 0.621 |
| Averaged Word2Vec + LogReg | 0.860 | 0.797–0.909 | 0.875 | 0.874 | 0.718 |
| BiLSTM + attention (Word2Vec init) | 0.861 ± 0.018 | 0.769–0.903 | 0.882 | 0.886 | 0.683 |
| Transformer from scratch | 0.752 ± 0.041 | 0.702–0.836 | 0.822 | 0.821 | 0.374 |
| Fine-tuned AfroXLMR / XLM-R | *run notebook 06* | | | | |

Published MasakhaNEWS weighted-F1 for Lingala (Adelani et al., 2023): NB 82.8, MLP 84.3, XLM-R base 87.9, AfroXLMR base 93.0.

What we learned:
* **Class weights** (+8 CV points) and **character n-grams** (half the fold-to-fold variance) matter most for the linear models. Naive Bayes with default smoothing on TF-IDF collapses to the majority class.
* Averaged Word2Vec and the BiLSTM improve the small **business** class, but paired bootstrap tests show they are **statistically tied** with TF-IDF on this test set.
* A Transformer **without pre-training** is worse than all classical models except NB, so the gains of pretrained Transformers come from pre-training.
* Most errors are **health or business stories with political framing** (e.g. "Parliament extends the health state of emergency"). Six test articles are misclassified by all five models, which points to label ambiguity (annotator kappa 0.56).

All experiment tables are in `results/experiment_logs/` and `results/comparison/`.

## 5. Web application

`app/app.py` is a Gradio app. The user pastes a Lingala headline and/or article, picks a model and gets the probability of each topic
(with the Lingala topic name), a warning when the model is unsure, and, for the TF-IDF model, the words that pushed the prediction.

```bash
python app/app.py                 # TF-IDF model only
TRANSFORMER_MODEL=YOUR-HF-USERNAME/lingala-news-topic-classifier python app/app.py   # also the fine-tuned Transformer
```

Deploying to Hugging Face Spaces:
1. Create a new Space (SDK: Gradio, hardware: free CPU).
2. Run `bash app/prepare_space.sh` and upload everything in `space_upload/` to the Space.
3. Optional: in the Space settings add the variable `TRANSFORMER_MODEL` with your model repository name (pushed from notebook 06).

## 6. How to reproduce

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/lingala-news-classification.git
cd lingala-news-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_preparation          # downloads and cleans the data into data/processed/
jupyter notebook                        # run notebooks 01 to 05 and 07 in order
```

Notebook 06 needs a GPU: open it in Google Colab (Runtime → T4 GPU), set `REPO_URL` in the first cell, run all, and copy the
downloaded `results/` folder back into the repository. Then re-run notebook 07.

All random seeds are fixed (`src/config.py`). Every experiment is appended to `results/experiment_logs/<family>.csv`, and the test
probabilities of every final model are saved in `results/predictions/`, so notebook 07 rebuilds every table without retraining.

## 7. Repository structure

```
app/                  Gradio web app, Space README, examples from the test set
data/raw/             MasakhaNEWS Lingala TSV files (train, dev, test)
data/processed/       Cleaned CSV splits (created by src/data_preparation.py)
models/               Saved TF-IDF model and Word2Vec vectors (the Transformer lives on the Hugging Face Hub)
notebooks/            01-07, see the table above
results/
  experiment_logs/    One CSV per model family, one row per experiment
  predictions/        Test-set probabilities of each final model
  misclassified/      Wrong test predictions, for the error analysis
  figures/            Plots per model
  comparison/         Final tables and figures from notebook 07
src/
  config.py           Paths, labels, seeds
  data_preparation.py Download, cleaning, quality checks
  data_loading.py     Load splits, label <-> id
  text_processing.py  Tokeniser and accent stripping
  classical_models.py TF-IDF pipelines (NB, LogReg, SVM)
  cross_validation.py Repeated stratified k-fold
  word_embeddings.py  Word2Vec training and document vectors
  rnn_models.py       BiLSTM/BiGRU + attention in PyTorch
  scratch_transformer.py   WordPiece tokeniser + small BERT from scratch
  transformer_training.py  Fine-tuning loop for Hugging Face models
  evaluation.py       Metrics, bootstrap intervals, significance test, error tables
  experiment_log.py   Experiment logging
  plots.py            Confusion matrices and learning curves
```

## 8. Acknowledgements and licences

* **Data:** MasakhaNEWS (Adelani et al., 2023), articles originally published by VOA Lingala. Use under the dataset's licence.
* **Pretrained models:** XLM-R (Conneau et al., 2020), AfroXLMR (Alabi et al., 2022) from the Hugging Face Hub.
* **Libraries:** scikit-learn, PyTorch, Hugging Face Transformers and Tokenizers, gensim, Gradio, pandas, matplotlib, seaborn.
* **What I implemented:** the data pipeline, all experiment code and notebooks, the RNN architecture, the from-scratch Transformer setup,
  the evaluation and error analysis, and the web app. Pretrained weights, the dataset and the libraries above are existing resources.
* AI assistance (Claude) was used to help write and debug code and documentation; I tested and verified the code and results.

## References

* Adelani, D. I., et al. (2023). MasakhaNEWS: News Topic Classification for African languages. *IJCNLP-AACL 2023*.
* Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning. *COLING 2022*.
* Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
* Eberhard, D. M., Simons, G. F., & Fennig, C. D. (Eds.) (2024). *Ethnologue: Languages of the World* (27th ed.). SIL International.
* Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.
* Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to Fine-Tune BERT for Text Classification? *CCL 2019*.
* Yang, Z., et al. (2016). Hierarchical Attention Networks for Document Classification. *NAACL 2016*.
