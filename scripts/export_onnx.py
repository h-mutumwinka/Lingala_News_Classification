"""Convert the fine-tuned model to ONNX (8-bit) so it can run in a web browser.

The static web app (app/static/index.html) uses Transformers.js, which runs ONNX
models directly in the browser. Dynamic 8-bit quantisation stores each weight
in 1 byte instead of 4, so the download shrinks from about 1.1 GB to about 280 MB.

Usage (Google Colab):
    !pip install -q onnx onnxruntime onnxscript
    !python scripts/export_onnx.py --model Heroine2/lingala-news-topic-classifier --output export
"""

import argparse
import os

import numpy as np
import torch
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LogitsOnly(torch.nn.Module):
    """Wrap the model so the exported graph returns a plain logits tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model_name: str, output_directory: str) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    onnx_directory = os.path.join(output_directory, "onnx")
    os.makedirs(onnx_directory, exist_ok=True)
    full_path = os.path.join(onnx_directory, "model.onnx")
    quantized_path = os.path.join(onnx_directory, "model_quantized.onnx")

    sample = tokenizer(["Ba Léopards babetami lisusu fimbu na CAN 2019", "FMI elobi nkita ekiti"],
                       padding=True, return_tensors="pt")
    torch.onnx.export(
        LogitsOnly(model),
        (sample["input_ids"], sample["attention_mask"]),
        full_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # Unsigned 8-bit, per-channel: the most accurate variant in our comparison on the test set
    quantize_dynamic(full_path, quantized_path, weight_type=QuantType.QUInt8, per_channel=True)

    # Check that the quantised model gives the same predictions as PyTorch
    with torch.no_grad():
        expected = model(**sample).logits.numpy()
    session = InferenceSession(quantized_path)
    actual = session.run(["logits"], {"input_ids": sample["input_ids"].numpy(),
                                      "attention_mask": sample["attention_mask"].numpy()})[0]
    print("PyTorch predictions:", expected.argmax(1), " ONNX 8-bit predictions:", actual.argmax(1))
    print("Largest logit difference:", float(np.abs(expected - actual).max()))
    for path in [full_path, quantized_path]:
        print(f"{path}: {os.path.getsize(path) / 1e6:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="export")
    arguments = parser.parse_args()
    export(arguments.model, arguments.output)
