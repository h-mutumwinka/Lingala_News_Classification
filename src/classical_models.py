"""Bag-of-words baselines: TF-IDF features with Naive Bayes, Logistic Regression or a linear SVM.

These models ignore word order. Each article becomes a long sparse vector with
one number per word (or character n-gram): its TF-IDF weight. TF-IDF is high for
a word that is frequent in this article but rare across all articles, so
"Léopards" or "Ebola" get large weights while "mpe" (and) or "ya" (of) get small ones.

All pipelines use only scikit-learn objects, so a saved model can be loaded in
the web app with joblib without importing this project.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src import config

# Letters of any alphabet (keeps é, è, ɛ, ɔ) or numbers. Same rule as src/text_processing.py
WORD_TOKEN_PATTERN = r"(?u)[^\W\d_]+|\d+"


def word_tfidf(ngram_range=(1, 1), strip_accents=None, min_df=1, sublinear_tf=True):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents=strip_accents,  # None (keep accents) or "unicode" (remove them)
        token_pattern=WORD_TOKEN_PATTERN,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,  # 1 + log(count): a word repeated 10 times is not 10x as important
    )


def char_tfidf(ngram_range=(2, 5), strip_accents=None, min_df=2):
    # "char_wb" builds character n-grams inside word boundaries.
    # Lingala is agglutinative: "bakonzi", "mokonzi", "bokonzi" share the stem
    # "konzi" (rule/leader) but have different noun-class prefixes. Character
    # n-grams let the model see that shared stem; whole-word features do not.
    return TfidfVectorizer(
        lowercase=True,
        strip_accents=strip_accents,
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )


def build_vectoriser(settings: dict):
    features = settings.get("features", "word")
    accents = "unicode" if settings.get("strip_accents") else None
    if features == "word":
        return word_tfidf(tuple(settings.get("word_ngrams", (1, 1))), accents, settings.get("min_df", 1))
    if features == "char":
        return char_tfidf(tuple(settings.get("char_ngrams", (2, 5))), accents)
    if features == "word+char":
        return FeatureUnion([
            ("word", word_tfidf(tuple(settings.get("word_ngrams", (1, 2))), accents, settings.get("min_df", 1))),
            ("char", char_tfidf(tuple(settings.get("char_ngrams", (2, 5))), accents)),
        ])
    raise ValueError(f"Unknown features: {features}")


def build_classifier(settings: dict):
    name = settings["classifier"]
    class_weight = "balanced" if settings.get("class_weight") else None
    if name == "majority":
        # Always predicts the class shares of the training data: the floor every model must beat
        return DummyClassifier(strategy="prior")
    if name == "naive_bayes":
        return MultinomialNB(alpha=settings.get("alpha", 1.0))
    if name == "complement_nb":
        # Complement NB (Rennie et al., 2003) is designed for unbalanced classes
        return ComplementNB(alpha=settings.get("alpha", 1.0))
    if name == "logistic_regression":
        return LogisticRegression(
            C=settings.get("C", 10.0),
            class_weight=class_weight,
            max_iter=5000,
            random_state=config.RANDOM_SEED,
        )
    if name == "linear_svm":
        # SVMs output scores, not probabilities. Calibration (Platt scaling with
        # internal 3-fold CV) turns the scores into probabilities for log loss and the app.
        return CalibratedClassifierCV(
            LinearSVC(C=settings.get("C", 1.0), class_weight=class_weight, random_state=config.RANDOM_SEED),
            cv=3,
        )
    raise ValueError(f"Unknown classifier: {name}")


def build_pipeline(settings: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", build_vectoriser(settings)),
        ("classifier", build_classifier(settings)),
    ])


def top_features_per_class(pipeline: Pipeline, number: int = 15) -> dict[str, list[str]]:
    """Words with the largest positive weight for each class (logistic regression only).

    Used in the error analysis and the web app to explain predictions.
    """
    vectoriser = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    names = vectoriser.get_feature_names_out()
    top = {}
    for class_id, label in enumerate(config.LABELS):
        weights = classifier.coef_[class_id]
        best = weights.argsort()[::-1][:number]
        top[label] = [names[i] for i in best]
    return top
