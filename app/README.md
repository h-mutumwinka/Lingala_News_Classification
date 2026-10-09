---
title: Lingala News Topic Classifier
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Classify Lingala news into politics, health, sports, business
---

# Lingala News Topic Classifier

Paste a Lingala news article and get its topic: politics, health, sports or business.
Trained on the Lingala part of MasakhaNEWS (Adelani et al., 2023).

Source code, experiments and report: https://github.com/YOUR-GITHUB-USERNAME/lingala-news-classification

To use the fine-tuned Transformer, set a Space variable `TRANSFORMER_MODEL` to the Hugging Face model
repository, for example `YOUR-HF-USERNAME/lingala-news-topic-classifier`.
