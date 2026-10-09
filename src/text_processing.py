"""Tokenisation and optional normalisation shared by the non-transformer models.

Transformers bring their own subword tokeniser, so they do not use this file.
"""

import re
import unicodedata

# Words of one or more letters (any alphabet, including é, ɛ, ɔ) or numbers.
# Apostrophes inside French loanwords such as "l'ONU" split into "l" and "onu".
TOKEN_PATTERN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def strip_accents(text: str) -> str:
    """Remove accent marks: "bokolongono" stays, "procès" -> "proces".

    Lingala writers mark tone (and French loanwords keep accents)
    inconsistently, so the same word can appear with and without accents.
    Stripping them is one of our preprocessing experiments.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


def normalise(text: str, lowercase: bool = True, remove_accents: bool = False) -> str:
    if lowercase:
        text = text.lower()
    if remove_accents:
        text = strip_accents(text)
    return text


def tokenise(text: str, lowercase: bool = True, remove_accents: bool = False) -> list[str]:
    return TOKEN_PATTERN.findall(normalise(text, lowercase, remove_accents))
