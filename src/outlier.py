# src/outlier.py
import numpy as np
from typing import Iterable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from .preprocess import preprocess_series, build_vectorizer

try:
    # koristimo ako postoji; nije obavezno
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None


def _to_dense(X) -> np.ndarray:
    """
    Pretvori features u gusti niz:
      - Ako je već ndarray, samo vrati.
      - Ako je scipy.sparse, uradi .toarray().
    """
    if isinstance(X, np.ndarray):
        return X
    if sp is not None and sp.issparse(X):
        return X.toarray()
    toarr = getattr(X, "toarray", None)
    if callable(toarr):
        return toarr()
    return np.asarray(X)


def anomaly_report(
    texts: Iterable[str],
    labels: Iterable[str],
    features: str = "tfidf",
    random_state: int = 42,
    contamination: float = 0.05,
):
    """
    Izračunaj 'novitet' / outlier score po poruci pomoću IsolationForest-a.
    Viši skor => 'čudniji' uzorak (potencijalno novi tip spama).

    Args:
        texts: lista/serija sirovih tekstova.
        labels: lista/serija labela (ne koristi se u učenju, samo je prosleđuješ radi kasnije analize).
        features: 'tfidf' | 'bow' | 'w2v' (u skladu sa build_vectorizer).
        random_state: seed za reproduktivnost.
        contamination: očekivani udeo outliera (0..0.5).

    Returns:
        np.ndarray oblika (n_samples,) sa outlier skorom po uzorku

    """
    # 1) čišćenje teksta
    X_clean = preprocess_series(texts)

    # 2) vektorization (tfidf/bow -> sparse; w2v -> dense)
    vec = build_vectorizer(features)
    Xv = vec.fit_transform(X_clean)

    # 3) IsolationForest radi nad gustim nizom
    X_dense = _to_dense(Xv).astype(np.float32, copy=False)

    # 4) treniraj IF i izračunaj skorove (negativan score_samples -> veće je 'čudnije')
    iso = IsolationForest(
        random_state=random_state,
        contamination=contamination,
        n_jobs=-1,
    )
    iso.fit(X_dense)
    scores = -iso.score_samples(X_dense)  # OK: veće => anomalnije

    return scores
