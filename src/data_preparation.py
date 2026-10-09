"""Download (if needed), clean and save the Lingala MasakhaNEWS splits.

Run from the project root:

    python -m src.data_preparation

We keep the official train / dev / test splits from MasakhaNEWS instead of
re-splitting, so our scores can be compared with the published baselines
(Adelani et al., 2023, Table 3).
"""

import re
import unicodedata
import urllib.request

import pandas as pd

from src import config

# --------------------------------------------------------------- download


def download_raw_splits() -> None:
    """Fetch the three TSV files if they are not in data/raw yet.

    We try Hugging Face first and fall back to the Masakhane GitHub mirror.
    Both hold the same 608 / 87 / 175 articles.
    """
    missing = [path for path in config.RAW_SPLIT_FILES.values() if not path.exists()]
    if not missing:
        return

    config.RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        dataset = load_dataset(config.HUGGING_FACE_DATASET, config.HUGGING_FACE_CONFIG)
        label_names = dataset["train"].features["label"].names
        for split_name, path in config.RAW_SPLIT_FILES.items():
            frame = dataset[split_name].to_pandas()
            frame["category"] = frame["label"].map(lambda i: label_names[i])
            frame[["category", "headline", "text", "url"]].to_csv(path, sep="\t", index=False)
        print("Downloaded MasakhaNEWS (lin) from Hugging Face")
        return
    except Exception as error:  # no internet access to HF, or `datasets` missing
        print(f"Hugging Face download failed ({error}); trying GitHub")

    github_names = {"train": "train", "validation": "dev", "test": "test"}
    for split_name, path in config.RAW_SPLIT_FILES.items():
        url = config.GITHUB_RAW_URL.format(split=github_names[split_name])
        urllib.request.urlretrieve(url, path)
    print("Downloaded MasakhaNEWS (lin) from GitHub")


def load_raw_split(split_name: str) -> pd.DataFrame:
    return pd.read_csv(config.RAW_SPLIT_FILES[split_name], sep="\t")


# --------------------------------------------------------------- cleaning

# Typographic characters that mean the same as a plain ASCII character.
FANCY_CHARACTERS = {
    "’": "'",  # right single quote, used as apostrophe: "ya l’ONU"
    "‘": "'",
    "“": '"',
    "”": '"',
    "«": '"',  # French guillemets « »
    "»": '"',
    "…": "...",
    "–": "-",
    "—": "-",
    " ": " ",  # non-breaking space
    "​": "",  # zero-width space
}


def normalise_unicode(text: str) -> str:
    # NFC joins a letter and a separate accent into one character, so "é"
    # is always stored the same way. Lingala uses accents for tone and
    # French loanwords keep theirs, so we keep the accents themselves.
    return unicodedata.normalize("NFC", text)


def replace_fancy_characters(text: str) -> str:
    for fancy, plain in FANCY_CHARACTERS.items():
        text = text.replace(fancy, plain)
    return text


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_odd_symbols(text: str) -> str:
    # A few articles contain stray symbols from the web page (music notes,
    # Arabic/Hebrew combining marks). Keep letters, digits, punctuation and spaces.
    return "".join(
        character
        for character in text
        if unicodedata.category(character)[0] in {"L", "N", "P", "Z"}
        or character in "$%&+=<>/"
    )


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = str(text)
    text = normalise_unicode(text)
    text = replace_fancy_characters(text)
    text = remove_urls(text)
    text = remove_odd_symbols(text)
    return collapse_whitespace(text)


def build_content(headline: str, text: str) -> str:
    # The headline is a short summary of the topic, so it is a strong signal.
    # A full stop keeps it a separate sentence for the tokenisers.
    headline = headline.rstrip(".!? ")
    return f"{headline}. {text}"


def clean_split(raw: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.DataFrame()
    cleaned["category"] = raw["category"].str.strip().str.lower()
    cleaned["headline"] = raw["headline"].apply(clean_text)
    cleaned["text"] = raw["text"].apply(clean_text)
    cleaned["content"] = [
        build_content(headline, text)
        for headline, text in zip(cleaned["headline"], cleaned["text"])
    ]
    cleaned["url"] = raw["url"]
    return cleaned


# ---------------------------------------------------------- quality checks


def check_labels(splits: dict[str, pd.DataFrame]) -> None:
    for split_name, frame in splits.items():
        unknown = set(frame["category"]) - set(config.LABELS)
        if unknown:
            raise ValueError(f"Unknown labels in {split_name}: {unknown}")


def check_overlap_between_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Make sure no article appears in two splits (that would leak test data)."""
    names = list(splits)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            shared_urls = set(splits[first]["url"]) & set(splits[second]["url"])
            shared_texts = set(splits[first]["text"]) & set(splits[second]["text"])
            print(
                f"{first} vs {second}: {len(shared_urls)} shared URLs, "
                f"{len(shared_texts)} identical texts"
            )


def print_split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = pd.DataFrame(
        {name: frame["category"].value_counts() for name, frame in splits.items()}
    ).reindex(config.LABELS)
    summary.loc["total"] = summary.sum()
    print(summary)
    return summary


def main() -> None:
    download_raw_splits()

    splits = {name: clean_split(load_raw_split(name)) for name in config.RAW_SPLIT_FILES}

    check_labels(splits)
    check_overlap_between_splits(splits)
    print_split_summary(splits)

    config.PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    splits["train"].to_csv(config.TRAIN_FILE, index=False)
    splits["validation"].to_csv(config.VALIDATION_FILE, index=False)
    splits["test"].to_csv(config.TEST_FILE, index=False)
    print(f"Saved cleaned splits to {config.PROCESSED_DATA_DIRECTORY}")


if __name__ == "__main__":
    main()
