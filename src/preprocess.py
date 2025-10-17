import re
import html as htmlmod
import unicodedata
from typing import Iterable
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MAIL_RE  = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")
NUM_RE   = re.compile(r"\b\d+([.,]\d+)*\b")
TAG_RE   = re.compile(r"<[^>]+>")  # brisanje HTML tagova
WS_RE    = re.compile(r"\s+")


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1) decode HTML, skini tagove
    text = htmlmod.unescape(text)
    text = TAG_RE.sub(" ", text)

    # 2) normalizacija, lowercase, ukloni dijakritike
    text = _strip_accents(text)
    text = text.lower()

    # 3) normalizacije tokova
    text = URL_RE.sub(" URL ", text)
    text = MAIL_RE.sub(" EMAIL ", text)
    text = NUM_RE.sub(" NUM ", text)

    # 4) ukloni ostatke “smeća”: sve što nije slovo/broj/razmak
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 5) zbij višestruke razmake
    text = WS_RE.sub(" ", text).strip()
    return text


def preprocess_series(texts: Iterable[str]) -> pd.Series:
    """Čišćenje niza tekstova (Series ili list)."""
    s = pd.Series(texts, dtype="string")
    return s.fillna("").map(clean_text)


def _tfidf_vectorizer():
    # strong baseline: unigrams+bigrams, filtriranje retkih/čestih, TF-idf sa sublinear TF
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
        strip_accents="unicode",
    )


def _bow_vectorizer():
    return CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        stop_words="english",
        strip_accents="unicode",
    )


def _svd_semantic():
    """LSA: TF-IDF -> SVD -> Normalizer (servira kao 'w2v' varijanta bez eksternih zavisnosti)."""
    return Pipeline(
        steps=[
            ("tfidf", _tfidf_vectorizer()),
            ("svd", TruncatedSVD(n_components=300, random_state=42)),
            ("norm", Normalizer(copy=False)),
        ]
    )


def build_vectorizer(features: str):
    """
    Vrati sklearn transformer za step 'vec' u pipeline-u.
    - 'tfidf' : TfidfVectorizer sa ngramima
    - 'bow'   : CountVectorizer sa ngramima
    - 'w2v'   : LSA (TF-IDF + TruncatedSVD) kao semantički embedding
    """
    f = (features or "tfidf").lower()
    if f == "tfidf":
        return _tfidf_vectorizer()
    if f == "bow":
        return _bow_vectorizer()
    if f == "w2v":
        return _svd_semantic()
    raise ValueError(f"Unknown features='{features}'. Use 'tfidf' | 'bow' | 'w2v'.")
