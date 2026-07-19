from __future__ import annotations

import re
import unicodedata
from typing import Iterable

DEFAULT_STOPWORDS = {
    "acaba", "ama", "aslında", "az", "bazı", "belki",
    "ben", "beni", "benim", "bir", "birkaç", "biz",
    "bu", "bunu", "çok", "da", "daha", "de", "diye",
    "en", "gibi", "hem", "hep", "her", "için", "ile",
    "ise", "ki", "mı", "mi", "mu", "mü", "nasıl",
    "ne", "neden", "o", "olan", "olarak", "oldu",
    "olur", "siz", "şu", "ve", "veya"
}

_TURKISH_LOWER = str.maketrans({
    "I": "ı",
    "İ": "i"
})


def normalize_text(text: str) -> str:

    value = unicodedata.normalize("NFKC", str(text))
    value = value.translate(_TURKISH_LOWER).lower()

    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"www\.\S+", " ", value)
    value = re.sub(r"\S+@\S+", " ", value)

    value = re.sub(r"[^0-9a-zçğıöşü\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def tokenize_text(
    text: str,
    remove_stopwords: bool = True,
    min_length: int = 2,
) -> list[str]:

    tokens = normalize_text(text).split()

    if remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in DEFAULT_STOPWORDS
        ]

    tokens = [
        token
        for token in tokens
        if len(token) >= min_length
    ]

    return tokens


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
) -> str:

    tokens = tokenize_text(
        text,
        remove_stopwords=remove_stopwords,
    )

    return " ".join(tokens)


def preprocess_corpus(
    texts: Iterable[str],
) -> list[str]:

    return [
        preprocess_text(text)
        for text in texts
    ]