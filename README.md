# Lingala News Topic Classification

Automatic topic classification of **Lingala** news articles into **politics, health, sports and business**, using the
Lingala part of [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews) (Adelani et al., 2023).

ALU · NLP and Language Technologies · Summative project (Option 3: Text Classification for an African Language)
Author: Joseph Nishimwe

| Resource | Link |
|---|---|
| GitHub repository | https://github.com/h-mutumwinka/Lingala_News_Classification |
| Live web app | https://h-mutumwinka.github.io/Lingala_News_Classification/ |
| Fine-tuned model | https://huggingface.co/Heroine2/lingala-news-topic-classifier |
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

Key findings from the EDA (Part 1 of the notebook): strong class imbalance; 53% of the training vocabulary occurs only once and about
7% of test tokens are unseen in training; frequent French code-mixing (*ba stocks*, *procès*, *Léopards*); inconsistent accents and
spellings (*président/president*, *Tshisekedi/Tshiskedi*); articles concentrated in 2018–2023 (Ebola, COVID-19).

## 3. Methodology

Everything runs from **one notebook, `notebooks/lingala_news_classification.ipynb`**, organised in seven parts.
The reusable code (models, training loops, metrics) lives in `src/`.

| Part | What it does |
|---|---|
| 1 | Exploratory data analysis |
| 2 | Majority class, TF-IDF + Naive Bayes / Logistic Regression / linear SVM; feature, input and preprocessing experiments; 5-fold x 3 cross-validation |
| 3 | Word2Vec trained on the training articles; averaged document vectors + logistic regression |
| 4 | BiLSTM / BiGRU with attention: random vs Word2Vec embeddings, frozen vs fine-tuned, class weights |
| 5 | Small BERT-style Transformer with its own WordPiece tokeniser, no pre-training (control) |
| 6 | Fine-tuning XLM-R base vs AfroXLMR base; truncation and class-weight ablations (needs a GPU) |
| 7 | Final comparison, bootstrap confidence intervals, significance tests, error analysis |

Evaluation protocol:
* Train on `train`, tune on `validation`, use `test` **once** per final model.
* Main metric **macro-F1** (all topics weigh the same). We also report weighted-F1 (the metric of the MasakhaNEWS paper), accuracy,
  macro precision/recall and per-class F1.
* Because the validation set is tiny (8 business articles), bag-of-words settings are chosen by 5-fold cross-validation repeated 3 times;
  neural models are retrained with 3 seeds (mean ± std); test scores come with 95% bootstrap intervals and paired bootstrap tests.

## 4. Results

Test set (175 articles), each final model evaluated once. Neural models: mean ± std over 3 seeds (Colab T4 GPU run).

| Model | Macro-F1 | 95% bootstrap CI | Weighted-F1 | Accuracy | F1 business |
|---|---|---|---|---|---|
| Majority class | 0.182 | 0.165–0.196 | 0.416 | 0.571 | 0.000 |
| TF-IDF + Complement NB | 0.673 | 0.609–0.738 | 0.773 | 0.811 | 0.111 |
| TF-IDF (word + char) + linear SVM | 0.822 | 0.747–0.884 | 0.852 | 0.857 | 0.621 |
| Averaged Word2Vec + LogReg | 0.860 | 0.797–0.909 | 0.875 | 0.874 | 0.718 |
| BiLSTM + attention (Word2Vec init) | 0.859 ± 0.026 | 0.822–0.935 | 0.888 | 0.893 | 0.661 |
| Transformer from scratch | 0.824 ± 0.041 | 0.703–0.853 | 0.876 | 0.880 | 0.563 |
| **Fine-tuned AfroXLMR (deployed)** | **0.911 ± 0.014** | **0.866–0.963** | **0.929** | **0.930** | **0.788** |

Published MasakhaNEWS weighted-F1 for Lingala (Adelani et al., 2023): NB 82.8, MLP 84.3, XLM-R base 87.9, AfroXLMR base 93.0.

What we learned:
* **Pre-training is what makes Transformers work:** the same kind of encoder scores 0.824 from scratch and 0.911 when fine-tuned from AfroXLMR (paired bootstrap vs TF-IDF: p = 0.002).
* **Pre-training on related African languages beats general multilingual pre-training:** AfroXLMR beats XLM-R by 4.5 validation macro-F1 points with the identical tokeniser, although neither saw Lingala.
* For the classical models, **class weights** (+8 CV points) and **character n-grams** (half the fold-to-fold variance) matter most. Naive Bayes with default smoothing on TF-IDF collapses to the majority class.
* Most remaining errors are **business or health stories with political framing** (e.g. "Parliament extends the health state of emergency"). Three test articles are misclassified by all six models, which points to label ambiguity (annotator kappa 0.56).

All experiment tables are in `results/experiment_logs/` and `results/comparison/`.

## 5. Web application

**Live app: https://h-mutumwinka.github.io/Lingala_News_Classification/** (hosted for free on GitHub Pages from the `docs/` folder).

The user pastes a Lingala headline and/or article, presses **Classify**, and gets the predicted topic with the probability of each topic
(with its Lingala name), a warning when the model is unsure or the input is very short, and six real test-set examples to try.

How the app connects to the model:
1. The fine-tuned AfroXLMR model is stored on the Hugging Face Hub (`Heroine2/lingala-news-topic-classifier`).
2. `scripts/export_onnx.py` converts it to ONNX and quantises the weights to 8 bits (about 1.1 GB to about 280 MB).
3. `docs/index.html` loads [Transformers.js](https://huggingface.co/docs/transformers.js), downloads the tokenizer and the 8-bit model once
   (then cached by the browser), builds the input exactly as in training (`headline. text`, truncated to 512 tokens), runs the model
   in the browser and applies a softmax to the four logits. No server is needed and the text never leaves the user's computer.

A short Gradio version (`app/app.py`, about 50 lines) can also be run locally:

```bash
pip install -r requirements.txt
TRANSFORMER_MODEL=Heroine2/lingala-news-topic-classifier python app/app.py
```

## 6. How to reproduce

```bash
git clone https://github.com/h-mutumwinka/Lingala_News_Classification.git
cd Lingala_News_Classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_preparation          # downloads and cleans the data into data/processed/
jupyter notebook notebooks/lingala_news_classification.ipynb
```

The easiest way is Google Colab: open the notebook from GitHub, choose **Runtime → Change runtime type → T4 GPU** and
**Run all** (about 1.5 hours). Part 6 needs the GPU. At the end the notebook downloads `results.zip`; unzip it into the repository and commit.

All random seeds are fixed (`src/config.py`). Every experiment is appended to `results/experiment_logs/<family>.csv`, and the test
probabilities of every final model are saved in `results/predictions/`, so Part 7 rebuilds every table without retraining.

## 7. Repository structure

```
app/                  Short local Gradio app (app.py)
docs/                 Static web app served by GitHub Pages
scripts/              ONNX export for the browser app
data/raw/             MasakhaNEWS Lingala TSV files (train, dev, test)
data/processed/       Cleaned CSV splits (created by src/data_preparation.py)
models/               Saved TF-IDF model and Word2Vec vectors (the Transformer lives on the Hugging Face Hub)
notebooks/            lingala_news_classification.ipynb (Parts 1-7)
results/
  experiment_logs/    One CSV per model family, one row per experiment
  predictions/        Test-set probabilities of each final model
  misclassified/      Wrong test predictions, for the error analysis
  figures/            Plots per model
  comparison/         Final tables and figures from Part 7
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
