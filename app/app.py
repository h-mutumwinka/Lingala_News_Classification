"""Simple local web app for the Lingala news topic classifier (Gradio).

The public app is the static page in docs/ (GitHub Pages). This file is a
short local alternative: it loads the fine-tuned AfroXLMR model and shows the
probability of each topic.

Run from the project root:   python app/app.py
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The fine-tuned model on the Hugging Face Hub (or a local folder with the same files)
MODEL = os.environ.get("TRANSFORMER_MODEL", "Heroine2/lingala-news-topic-classifier")
LINGALA = {"politics": "politiki", "health": "bokolongono", "sports": "masano", "business": "mombongo"}

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()


def classify(headline: str, text: str) -> dict[str, float]:
    # Same input format as in training: "headline. text"
    content = f"{headline.strip().rstrip('.!? ')}. {text.strip()}" if headline.strip() and text.strip() else (headline or text).strip()
    if not content:
        return {}
    # Tokenise and cut to 512 sub-word tokens, as in training
    inputs = tokenizer(content, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probabilities = torch.softmax(logits, dim=0)
    return {f"{label} ({LINGALA[label]})": float(probabilities[i]) for i, label in model.config.id2label.items()}


demo = gr.Interface(
    fn=classify,
    inputs=[gr.Textbox(label="Headline"), gr.Textbox(label="Article text", lines=10)],
    outputs=gr.Label(num_top_classes=4, label="Predicted topic"),
    title="Lingala News Topic Classifier",
    description="Paste a Lingala news article. The fine-tuned AfroXLMR model predicts politics, health, sports or business.",
    examples=[
        ["Ba Léopards babetami lisusu fimbu na CAN 2019", ""],
        ["FMI elobi nkita ekiti moke na RDC", ""],
        ["Bakoni ba COVID-19 bakomi 30 na Kinshasa", ""],
    ],
)

if __name__ == "__main__":
    demo.launch()
