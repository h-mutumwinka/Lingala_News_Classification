"""Word2Vec embeddings trained on our own Lingala training articles.

General-purpose Lingala vectors are scarce (fastText's Wikipedia vectors are
trained on the small Lingala Wikipedia, a different domain). We therefore train
in-domain vectors ourselves with gensim (Mikolov et al., 2013) on the 608
training articles only (about 110k words). Validation and test text is never used, so
nothing leaks from the evaluation sets.

The vectors are used in two ways:
1. Averaged into one document vector, fed to logistic regression.
2. As the starting weights of the embedding layer of the BiLSTM / BiGRU.
"""

import numpy as np
from gensim.models import Word2Vec

from src import config
from src.text_processing import tokenise


def train_word2vec(texts, settings: dict) -> Word2Vec:
    sentences = [tokenise(text, remove_accents=settings.get("strip_accents", False)) for text in texts]
    return Word2Vec(
        sentences=sentences,
        vector_size=settings.get("dimensions", 100),
        window=settings.get("window", 5),
        min_count=settings.get("min_count", 2),
        sg=1 if settings.get("skip_gram", True) else 0,  # skip-gram is usually better for rare words
        negative=10,
        epochs=settings.get("epochs", 30),  # many passes because the corpus is tiny
        seed=config.RANDOM_SEED,
        workers=1,  # one worker makes training reproducible
    )


def average_document_vectors(texts, word2vec: Word2Vec, strip_accents: bool = False) -> np.ndarray:
    """One vector per article: the mean of the vectors of its known words."""
    vectors = np.zeros((len(texts), word2vec.vector_size), dtype=np.float32)
    for row, text in enumerate(texts):
        known = [w for w in tokenise(text, remove_accents=strip_accents) if w in word2vec.wv]
        if known:
            vectors[row] = np.mean([word2vec.wv[w] for w in known], axis=0)
    return vectors


def nearest_words(word2vec: Word2Vec, words: list[str], number: int = 6) -> dict[str, list[str]]:
    """Sanity check: do the vectors put related words close together?"""
    return {
        word: [neighbour for neighbour, _ in word2vec.wv.most_similar(word, topn=number)]
        for word in words
        if word in word2vec.wv
    }


def embedding_matrix_for_vocabulary(vocabulary: dict[str, int], word2vec: Word2Vec, seed: int) -> tuple[np.ndarray, float]:
    """Rows in the order of our vocabulary. Words without a vector start random."""
    generator = np.random.default_rng(seed)
    matrix = generator.normal(0, 0.1, (len(vocabulary), word2vec.vector_size)).astype(np.float32)
    matrix[0] = 0  # padding token
    found = 0
    for word, index in vocabulary.items():
        if word in word2vec.wv:
            matrix[index] = word2vec.wv[word]
            found += 1
    return matrix, found / len(vocabulary)
