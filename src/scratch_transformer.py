"""A small Transformer trained from scratch on our 608 articles (no pre-training).

This is a control experiment: same kind of architecture as the pretrained models
(BERT-style encoder with self-attention), but every weight starts random and the
only text it ever sees is our training set. Comparing it with fine-tuned
pretrained models isolates the effect of pre-training.

It is small enough (about 1-2M parameters) to train on a CPU in minutes.
"""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from src import config


def train_wordpiece_tokeniser(texts, vocabulary_size: int, directory: Path) -> PreTrainedTokenizerFast:
    """Learn a WordPiece subword vocabulary (Wu et al., 2016) from the training articles."""
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocabulary_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.train_from_iterator(list(texts), trainer)

    # Add [CLS] at the start and [SEP] at the end of every article, as BERT expects
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )
    tokenizer.decoder = decoders.WordPiece()

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_max_length=512,
    )
    directory.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(directory)
    return wrapped


def build_scratch_model(tokenizer: PreTrainedTokenizerFast, settings: dict) -> BertForSequenceClassification:
    bert_config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=settings["hidden_size"],
        num_hidden_layers=settings["layers"],
        num_attention_heads=settings["heads"],
        intermediate_size=settings["hidden_size"] * 4,
        max_position_embeddings=settings["max_tokens"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=len(config.LABELS),
        id2label=dict(enumerate(config.LABELS)),
        label2id={label: i for i, label in enumerate(config.LABELS)},
    )
    return BertForSequenceClassification(bert_config)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
