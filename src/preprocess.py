import re
import html as htmlmod
import unicodedata
from typing import Iterable
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    nltk.download("wordnet", quiet=True)
    _LEMM = WordNetLemmatizer()
except Exception:
    _LEMM = None

# ===== Regex šabloni =====
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MAIL_RE  = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")
NUM_RE   = re.compile(r"\b\d+(?:[.,]\d+)*\b")
TAG_RE   = re.compile(r"<[^>]+>")           # HTML tagovi
WS_RE    = re.compile(r"\s+")
NON_ALNUM = re.compile(r"[^a-z0-9\s]")      # sve osim slova, brojeva i razmaka

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _normalize_placeholders(s: str) -> str:
    s = URL_RE.sub(" <url> ", s)
    s = MAIL_RE.sub(" <email> ", s)
    s = NUM_RE.sub(" <number> ", s)
    return s

def _lemmatize(s: str) -> str:
    if _LEMM is None:
        return s
    return " ".join(_LEMM.lemmatize(w) for w in s.split())

def clean_text(text: str) -> str:
    """Osnovno čišćenje + normalizacija placeholdera (URL/EMAIL/NUMBER)."""
    if not isinstance(text, str):
        return ""
    # 1) decode HTML, skini tagove
    text = htmlmod.unescape(text)
    text = TAG_RE.sub(" ", text)

    # 2) ukloni dijakritike, spusti na lowercase
    text = _strip_accents(text).lower()

    # 3) normalizuj specifične tokene
    text = _normalize_placeholders(text)

    # 4) ukloni sve što nije [a-z0-9/razmak]
    text = NON_ALNUM.sub(" ", text)

    # 5) zbij razmake
    text = WS_RE.sub(" ", text).strip()
    return text

def preprocess_series(texts: Iterable[str]) -> pd.Series:
    s = pd.Series(texts, dtype="string").fillna("")
    s = s.map(clean_text)
    s = s.map(_lemmatize)  # bezbedno je i ako _LEMM nije dostupan
    return s

# ===== Vectorizeri =====
def _tfidf_vectorizer():
    # Unigram+bigram, filtriranje retkih/čestih, TF-IDF sa sublinear TF
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

def build_vectorizer(kind: str = "tfidf"):
    """
    Vraća transformer za 'vec' korak u sklearn Pipeline-u.
    - 'bow'   : CountVectorizer sa ngramima
    - 'tfidf' : TfidfVectorizer sa ngramima
    - 'w2v'   : MeanEmbeddingVectorizer (prosečni W2V embedding po dokumentu)
    """
    k = (kind or "tfidf").lower()
    if k == "bow":
        return _bow_vectorizer()
    if k == "tfidf":
        return _tfidf_vectorizer()
    if k == "w2v":
        from .w2v_vectorizer import MeanEmbeddingVectorizer
        return MeanEmbeddingVectorizer(size=100, window=5, min_count=2)
    raise ValueError(f"Unknown features='{kind}'. Use 'tfidf' | 'bow' | 'w2v'.")
