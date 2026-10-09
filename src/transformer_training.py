"""Fine-tuning (or training from scratch) a Transformer encoder for topic classification.

Data flow:
    article text -> subword tokeniser (SentencePiece / WordPiece) -> token ids (max 512)
    -> Transformer encoder (self-attention layers) -> vector of the first token (<s> / [CLS])
    -> classification head (dense + tanh + linear) -> 4 logits -> softmax

Fine-tuning updates *all* weights of the pretrained encoder plus the new, randomly
initialised classification head, with a small learning rate (2e-5 for base models)
so the knowledge from pre-training is not destroyed (Devlin et al., 2019; Sun et al., 2019).

Pretrained checkpoints are downloaded from the Hugging Face Hub, so run the
fine-tuning notebook on Google Colab with a GPU.
"""

import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from src import config, data_loading
from src.evaluation import compute_scores
from src.reproducibility import set_seed

MAX_TOKENS = 512

# Sun et al. (2019): for long texts, keeping the beginning and the end works better
# than keeping only the beginning. We keep the first 128 and the last 382 tokens.
HEAD_TOKENS = 128


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_maps() -> tuple[dict[int, str], dict[str, int]]:
    id2label = dict(enumerate(config.LABELS))
    return id2label, {label: i for i, label in id2label.items()}


def shorten_token_ids(token_ids: list[int], strategy: str, max_tokens: int) -> list[int]:
    room = max_tokens - 2  # leave space for the start and end tokens
    if len(token_ids) <= room:
        return token_ids
    if strategy == "head":
        return token_ids[:room]
    head = min(HEAD_TOKENS, room // 2)
    return token_ids[:head] + token_ids[-(room - head):]


def encode_articles(tokenizer: PreTrainedTokenizerBase, data: pd.DataFrame, settings: dict) -> list[dict]:
    label_ids = data_loading.labels_to_ids(data["category"]) if "category" in data else np.zeros(len(data), int)
    max_tokens = settings.get("max_tokens", MAX_TOKENS)
    # Tokenising an empty string returns only the special tokens, e.g. [CLS] [SEP] for BERT
    # or <s> </s> for XLM-R. We put the first before the article and the rest after it.
    special = tokenizer("")["input_ids"]
    prefix, suffix = special[:1], special[1:]
    encoded = []
    for text, label_id in zip(data[config.TEXT_COLUMN], label_ids):
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        short = shorten_token_ids(token_ids, settings["strategy"], max_tokens)
        input_ids = prefix + short + suffix
        encoded.append({"input_ids": input_ids, "labels": int(label_id)})
    return encoded


def subword_statistics(tokenizer: PreTrainedTokenizerBase, texts) -> dict[str, float]:
    """Fertility = subword tokens per word. High fertility means the tokeniser
    does not know the language well and chops words into many small pieces."""
    words = sum(len(t.split()) for t in texts)
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    return {
        "tokens_per_word": round(sum(lengths) / words, 3),
        "share_truncated_at_512": round(float(np.mean(np.array(lengths) > MAX_TOKENS - 2)), 3),
    }


def build_data_loader(encoded, tokenizer, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    return DataLoader(
        encoded,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def build_pretrained_model(checkpoint: str) -> torch.nn.Module:
    id2label, label2id = label_maps()
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(config.LABELS), id2label=id2label, label2id=label2id
    )


def class_weight_tensor(train: pd.DataFrame) -> torch.Tensor:
    ids = data_loading.labels_to_ids(train["category"])
    counts = np.bincount(ids, minlength=len(config.LABELS))
    return torch.tensor(len(ids) / (len(config.LABELS) * counts), dtype=torch.float)


def predict_logits(model, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def train_model(model, train_loader, validation_loader, validation_ids, settings, class_weights=None):
    device = get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0.01)
    total_steps = len(train_loader) * settings["epochs"]
    # Warm-up: the learning rate grows during the first 10% of steps, so the random
    # classification head does not send large, noisy gradients into the pretrained layers
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = {"train_loss": [], "validation_loss": [], "validation_macro_f1": []}
    best_f1, best_state, patience_left = -1.0, None, settings["patience"]

    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_function(logits.float(), batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        scores = compute_scores(validation_ids, softmax(predict_logits(model, validation_loader, device)))
        history["train_loss"].append(total_loss / len(train_loader))
        history["validation_loss"].append(scores["log_loss"])
        history["validation_macro_f1"].append(scores["macro_f1"])
        print(f"Epoch {epoch}: train loss {history['train_loss'][-1]:.4f}, "
              f"validation loss {scores['log_loss']:.4f}, macro-F1 {scores['macro_f1']:.4f}")

        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience_left = settings["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return history


def run_experiment(settings: dict, train: pd.DataFrame, validation: pd.DataFrame, model=None, tokenizer=None):
    """Fine-tune settings["checkpoint"] (or train a model passed in) and return model, tokenizer, history."""
    set_seed(settings["seed"])
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(settings["checkpoint"])
    model = model or build_pretrained_model(settings["checkpoint"])

    train_loader = build_data_loader(encode_articles(tokenizer, train, settings), tokenizer,
                                     settings["batch_size"], shuffle=True, seed=settings["seed"])
    validation_loader = build_data_loader(encode_articles(tokenizer, validation, settings), tokenizer,
                                          settings["batch_size"] * 2, shuffle=False)
    validation_ids = data_loading.labels_to_ids(validation["category"])
    weights = class_weight_tensor(train) if settings.get("class_weight") else None

    history = train_model(model, train_loader, validation_loader, validation_ids, settings, weights)
    return model, tokenizer, history


def predict_probabilities(model, tokenizer, data: pd.DataFrame, settings: dict) -> np.ndarray:
    loader = build_data_loader(encode_articles(tokenizer, data, settings), tokenizer,
                               settings["batch_size"] * 2, shuffle=False)
    return softmax(predict_logits(model, loader, get_device()))


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
