"""Lingala news topic classifier: Gradio web app.

Paste a Lingala news article (or just a headline) and the app predicts whether it is
about politics, health, sports or business.

Two models can be used:
* "TF-IDF + linear SVM": the best bag-of-words model (notebook 02). Small, fast, always available.
  The app also shows which words pushed the prediction.
* "Fine-tuned Transformer": the best fine-tuned pretrained model (notebook 06). Loaded from the
  Hugging Face Hub repository named in the TRANSFORMER_MODEL environment variable, or from
  models/transformer_final if it exists locally.

Run locally from the project root:   python app/app.py
"""

import json
import os
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

APP_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIRECTORY.parent

LABELS = ["politics", "health", "sports", "business"]
LABEL_LINGALA = {"politics": "politiki", "health": "bokolongono", "sports": "masano", "business": "mombongo"}
DISPLAY = {label: f"{label} ({LABEL_LINGALA[label]})" for label in LABELS}

LINEAR_NAME = "TF-IDF + linear SVM (baseline)"
TRANSFORMER_NAME = "Fine-tuned Transformer"


def find_file(name: str) -> Path | None:
    for candidate in [APP_DIRECTORY / name, PROJECT_ROOT / "models" / name]:
        if candidate.exists():
            return candidate
    return None


# ------------------------------------------------------------------ models

linear_model = joblib.load(find_file("tfidf_classifier.joblib"))

TRANSFORMER_SOURCE = os.environ.get("TRANSFORMER_MODEL") or (
    str(PROJECT_ROOT / "models" / "transformer_final")
    if (PROJECT_ROOT / "models" / "transformer_final").exists()
    else None
)
_transformer = {}


def load_transformer():
    if "model" not in _transformer:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _transformer["tokenizer"] = AutoTokenizer.from_pretrained(TRANSFORMER_SOURCE)
        _transformer["model"] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_SOURCE).eval()
        _transformer["torch"] = torch
    return _transformer


def build_content(headline: str, text: str) -> str:
    """Same input format as in training: 'headline. text'."""
    headline, text = headline.strip(), text.strip()
    if headline and text:
        return f"{headline.rstrip('.!? ')}. {text}"
    return headline or text


def linear_probabilities(content: str) -> np.ndarray:
    return linear_model.predict_proba([content])[0]


def transformer_probabilities(content: str) -> np.ndarray:
    parts = load_transformer()
    torch = parts["torch"]
    encoded = parts["tokenizer"](content, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = parts["model"](**encoded).logits[0]
    return torch.softmax(logits, dim=0).numpy()


def linear_coefficients() -> np.ndarray:
    classifier = linear_model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        return classifier.coef_
    # CalibratedClassifierCV: average the coefficients of the SVMs trained on each fold
    return np.mean([c.estimator.coef_ for c in classifier.calibrated_classifiers_], axis=0)


COEFFICIENTS = linear_coefficients()
FEATURE_NAMES = linear_model.named_steps["tfidf"].get_feature_names_out()


def explain_linear(content: str, label_id: int, number: int = 8) -> str:
    """Words whose TF-IDF weight x model coefficient pushed most towards the predicted topic."""
    features = linear_model.named_steps["tfidf"].transform([content]).tocoo()
    contributions = {}
    for column, value in zip(features.col, features.data):
        name = FEATURE_NAMES[column]
        if not name.startswith("word__") and "__" in name:
            continue  # character n-grams are hard to read, so we show only word features
        contributions[name.split("__")[-1]] = value * COEFFICIENTS[label_id, column]
    top = sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:number]
    top = [(word, score) for word, score in top if score > 0]
    if not top:
        return "_No single word stands out; the decision comes from many small signals._"
    return "**Words that pushed towards this topic:** " + ", ".join(f"`{word}`" for word, _ in top)


def classify(headline: str, text: str, model_name: str):
    content = build_content(headline, text)
    if len(content.split()) < 3:
        return {}, "Please enter at least a headline or a few sentences in Lingala."

    use_transformer = model_name == TRANSFORMER_NAME and TRANSFORMER_SOURCE
    probabilities = transformer_probabilities(content) if use_transformer else linear_probabilities(content)
    label_id = int(np.argmax(probabilities))

    notes = [f"**Prediction:** {DISPLAY[LABELS[label_id]]} with probability {probabilities[label_id]:.0%} "
             f"({'Transformer' if use_transformer else 'TF-IDF + SVM'})."]
    if probabilities[label_id] < 0.6:
        notes.append("The model is unsure: the article may mix topics (for example a government decision about a disease).")
    if not use_transformer:
        notes.append(explain_linear(content, label_id))
    words = len(content.split())
    if words < 25:
        notes.append(f"_Only {words} words: short inputs give less evidence. The models were trained on full articles (median 165 words)._")

    return {DISPLAY[label]: float(p) for label, p in zip(LABELS, probabilities)}, "\n\n".join(notes)


# ---------------------------------------------------------------------- UI

examples_file = find_file("examples.json")
EXAMPLES = json.loads(examples_file.read_text()) if examples_file else []
model_choices = [LINEAR_NAME] + ([TRANSFORMER_NAME] if TRANSFORMER_SOURCE else [])

DESCRIPTION = """
Classifies **Lingala** news into **politics, health, sports or business**.
Trained on the Lingala part of [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews)
(Adelani et al., 2023): 608 Voice of America Lingala articles.

Paste a headline and/or article text in Lingala, choose a model, and press **Classify**.
"""

LIMITATIONS = """
**Limitations.** Only four topics are known, so an article about culture or religion will still be forced into one of them.
The training news comes from VOA Lingala and mostly covers 2018–2023 (Ebola, COVID-19, DRC politics), so newer topics and other
writing styles (social media, Brazzaville spelling) may be classified less reliably. Articles that mix topics, such as a
parliament vote on a health emergency, are the most common errors.
"""

with gr.Blocks(title="Lingala News Topic Classifier") as demo:
    gr.Markdown("# Lingala News Topic Classifier")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=3):
            headline_box = gr.Textbox(label="Headline (motó ya likambo)", lines=1)
            text_box = gr.Textbox(label="Article text (lisolo)", lines=10)
            model_box = gr.Radio(model_choices, value=model_choices[-1], label="Model")
            button = gr.Button("Classify", variant="primary")
        with gr.Column(scale=2):
            label_output = gr.Label(num_top_classes=4, label="Predicted topic")
            explanation_output = gr.Markdown()
    if EXAMPLES:
        gr.Examples(
            examples=[[e["headline"], e["text"], model_choices[-1]] for e in EXAMPLES],
            inputs=[headline_box, text_box, model_box],
            label="Examples from the MasakhaNEWS Lingala test set",
        )
        gr.Markdown("**True topics of the examples** (not shown to the model): "
                    + "; ".join(f"{i + 1}. {e['true_label']}" for i, e in enumerate(EXAMPLES)))
    gr.Markdown(LIMITATIONS)
    button.click(classify, [headline_box, text_box, model_box], [label_output, explanation_output])

if __name__ == "__main__":
    demo.launch()
