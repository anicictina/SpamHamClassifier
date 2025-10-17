# src/outlier.py
import numpy as np
from typing import Iterable
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD
from .preprocess import preprocess_series, build_vectorizer

def anomaly_report(
    texts: Iterable[str],
    labels: Iterable[str],
    features: str = "tfidf",       # 'tfidf' | 'bow' | 'w2v'
    random_state: int = 42,
    contamination: float = 0.05,
    n_components: int = 300        # broj SVD komponenti (za tfidf/bow)
):
    """
    Računa outlier/anomaly score po poruci (veći = sumnjivije).
    Ključno: za tfidf/bow NE pravimo gustu matricu, već radimo SVD (LSA) nad sparse matricom.
    Za 'w2v' vektori su već mali i gusti, pa SVD nije potreban.
    """
    # 1) čišćenje
    X_clean = preprocess_series(texts)

    # 2) vektorizacija
    vec = build_vectorizer(features)
    Xv = vec.fit_transform(X_clean)   # tfidf/bow: sparse, w2v: dense ndarray

    # 3) dim. redukcija samo ako je tfidf/bow (dakle, sparse)
    X_small = None
    if features.lower() in ("tfidf", "bow"):
        # TruncatedSVD radi direktno nad sparse matricom
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        # dobijamo mali gusti prikaz (N x n_components), float32
        X_small = svd.fit_transform(Xv).astype(np.float32, copy=False)
    else:
        # w2v već daje gustu malu matricu
        X_small = np.asarray(Xv, dtype=np.float32)

    # 4) IsolationForest nad malim gustim vektorima
    iso = IsolationForest(
        random_state=random_state,
        contamination=contamination,
        n_jobs=-1,
    )
    iso.fit(X_small)
    scores = -iso.score_samples(X_small)  # veći skor => anomalnije
    return scores
