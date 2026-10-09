"""Recurrent classifiers (BiLSTM / BiGRU) with attention pooling, in PyTorch.

Data flow for one article:

    words -> ids -> embedding layer (100-d per word, random or Word2Vec)
          -> bidirectional LSTM/GRU (reads left-to-right and right-to-left,
             2 x 128 = 256 numbers per word that describe the word in context)
          -> pooling: one 256-d vector for the whole article
               * "attention": a learned score per word, softmax over the article,
                 weighted sum of the word vectors (Bahdanau et al., 2015; Yang et al., 2016)
               * "last": just the final hidden states of both directions
          -> dropout -> linear layer -> 4 logits -> softmax -> class probabilities

Attention lets the model focus on topic words ("Léopards", "Ebola", "budget")
wherever they appear, which matters for articles of 150-500 words.
"""

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config, data_loading
from src.evaluation import compute_scores
from src.reproducibility import set_seed
from src.text_processing import tokenise

PAD_ID = 0
UNKNOWN_ID = 1

# ------------------------------------------------------------- vocabulary


def build_vocabulary(texts, min_count: int = 2, strip_accents: bool = False) -> dict[str, int]:
    counts = Counter(word for text in texts for word in tokenise(text, remove_accents=strip_accents))
    vocabulary = {"<pad>": PAD_ID, "<unk>": UNKNOWN_ID}
    for word, count in counts.most_common():
        if count >= min_count:
            vocabulary[word] = len(vocabulary)
    return vocabulary


def encode_texts(texts, vocabulary: dict[str, int], max_length: int, strip_accents: bool = False):
    ids = np.full((len(texts), max_length), PAD_ID, dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for row, text in enumerate(texts):
        tokens = tokenise(text, remove_accents=strip_accents)[:max_length]
        token_ids = [vocabulary.get(t, UNKNOWN_ID) for t in tokens] or [UNKNOWN_ID]
        ids[row, : len(token_ids)] = token_ids
        lengths[row] = len(token_ids)
    return torch.from_numpy(ids), torch.from_numpy(lengths)


# ------------------------------------------------------------------ model


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, states, mask):
        # states: (batch, words, hidden); mask: (batch, words), True for real words
        scores = self.context(torch.tanh(self.projection(states))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))  # padding gets zero weight
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return pooled, weights


class RecurrentClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, settings: dict, pretrained_embeddings: np.ndarray | None = None):
        super().__init__()
        dimensions = settings["embedding_dimensions"]
        self.embedding = nn.Embedding(vocabulary_size, dimensions, padding_idx=PAD_ID)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = not settings.get("freeze_embeddings", False)

        recurrent_layer = nn.LSTM if settings["cell"] == "lstm" else nn.GRU
        self.encoder = recurrent_layer(
            dimensions, settings["hidden_size"], batch_first=True, bidirectional=True
        )
        self.pooling_type = settings["pooling"]
        output_size = 2 * settings["hidden_size"]
        if self.pooling_type == "attention":
            self.attention = AttentionPooling(output_size)
        self.embedding_dropout = nn.Dropout(settings.get("embedding_dropout", 0.2))
        self.dropout = nn.Dropout(settings["dropout"])
        self.output = nn.Linear(output_size, len(config.LABELS))

    def forward(self, ids, lengths, return_attention: bool = False):
        embedded = self.embedding_dropout(self.embedding(ids))
        # Packing tells the RNN to stop at each article's real length (ignore padding)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_states, final = self.encoder(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(packed_states, batch_first=True, total_length=ids.shape[1])

        weights = None
        if self.pooling_type == "attention":
            pooled, weights = self.attention(states, ids != PAD_ID)
        else:  # last hidden state of the forward and backward directions
            hidden = final[0] if isinstance(final, tuple) else final
            pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)

        logits = self.output(self.dropout(pooled))
        return (logits, weights) if return_attention else logits


# --------------------------------------------------------------- training


def predict_probabilities(model, ids, lengths, batch_size: int = 64) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            logits = model(ids[start : start + batch_size], lengths[start : start + batch_size])
            outputs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(outputs)


def class_weights_from(train_ids: np.ndarray) -> torch.Tensor:
    # "balanced" weights: n_samples / (n_classes * count of class); business gets ~2.7, politics ~0.43
    counts = np.bincount(train_ids, minlength=len(config.LABELS))
    return torch.tensor(len(train_ids) / (len(config.LABELS) * counts), dtype=torch.float)


def train_recurrent_model(
    settings: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    pretrained_embeddings_fn=None,
):
    """Train one model with early stopping on validation macro-F1.

    pretrained_embeddings_fn(vocabulary) -> matrix, or None for random embeddings.
    Returns the model, the vocabulary and the training history.
    """
    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    accents = settings.get("strip_accents", False)

    vocabulary = build_vocabulary(train[config.TEXT_COLUMN], settings["min_count"], accents)
    train_x, train_len = encode_texts(train[config.TEXT_COLUMN], vocabulary, settings["max_length"], accents)
    val_x, val_len = encode_texts(validation[config.TEXT_COLUMN], vocabulary, settings["max_length"], accents)
    train_y = data_loading.labels_to_ids(train["category"])
    val_y = data_loading.labels_to_ids(validation["category"])

    embeddings = pretrained_embeddings_fn(vocabulary) if pretrained_embeddings_fn else None
    model = RecurrentClassifier(len(vocabulary), settings, embeddings)

    weights = class_weights_from(train_y) if settings.get("class_weight") else None
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["learning_rate"],
        weight_decay=settings.get("weight_decay", 0.0),
    )

    loader = DataLoader(
        TensorDataset(train_x, train_len, torch.from_numpy(train_y.copy())),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    history = {"train_loss": [], "validation_loss": [], "validation_macro_f1": []}
    best_f1, best_state, epochs_without_improvement = -1.0, None, 0

    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.0
        for batch_x, batch_len, batch_y in loader:
            optimiser.zero_grad()
            loss = loss_function(model(batch_x, batch_len), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops exploding gradients in RNNs
            optimiser.step()
            total += loss.item() * len(batch_y)

        val_probabilities = predict_probabilities(model, val_x, val_len)
        val_scores = compute_scores(val_y, val_probabilities)
        history["train_loss"].append(total / len(train_y))
        history["validation_loss"].append(val_scores["log_loss"])
        history["validation_macro_f1"].append(val_scores["macro_f1"])

        # Keep the epoch with the best validation macro-F1 (ties -> lower loss later is ignored)
        if val_scores["macro_f1"] > best_f1:
            best_f1 = val_scores["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings["patience"]:
                break

    model.load_state_dict(best_state)
    return model, vocabulary, history


def predict_dataframe(model, vocabulary, data: pd.DataFrame, settings: dict) -> np.ndarray:
    ids, lengths = encode_texts(
        data[config.TEXT_COLUMN], vocabulary, settings["max_length"], settings.get("strip_accents", False)
    )
    return predict_probabilities(model, ids, lengths)


def attention_for_text(model, vocabulary, text: str, settings: dict, top: int = 10) -> list[tuple[str, float]]:
    """Words the attention layer weighted most for one article (for error analysis)."""
    tokens = tokenise(text, remove_accents=settings.get("strip_accents", False))[: settings["max_length"]]
    ids, lengths = encode_texts([text], vocabulary, settings["max_length"], settings.get("strip_accents", False))
    model.eval()
    with torch.no_grad():
        _, weights = model(ids, lengths, return_attention=True)
    weights = weights[0, : len(tokens)].numpy()
    order = weights.argsort()[::-1][:top]
    return [(tokens[i], round(float(weights[i]), 3)) for i in order]
