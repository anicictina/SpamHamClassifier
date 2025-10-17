from typing import Iterable, List
import numpy as np
from gensim.models import Word2Vec

class MeanEmbeddingVectorizer:
    """
    Fit: trenira mali Word2Vec na korpusu.
    Transform: prosečan embedding po dokumentu.
    """
    def __init__(self, size: int = 100, window: int = 5, min_count: int = 2, workers: int = 4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def _tokenize(self, s: str) -> List[str]:
        return [t for t in s.split() if t]

    def fit(self, texts: Iterable[str], y=None):
        sentences = [self._tokenize(t) for t in texts]
        self.model = Word2Vec(sentences=sentences, vector_size=self.size,
                              window=self.window, min_count=self.min_count,
                              workers=self.workers, sg=1)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        vectors = []
        for t in texts:
            toks = self._tokenize(t)
            vecs = [self.model.wv[w] for w in toks if w in self.model.wv]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(self.size))
        return np.vstack(vectors)
