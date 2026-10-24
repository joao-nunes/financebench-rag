from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    """
    Okapi BM25 over a fixed collection of texts.

    Term weights are precomputed into a sparse (documents x vocabulary)
    matrix, so scoring a query is a sum over the columns of its terms.
    Positions in the index match positions in the texts passed to build().
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vectorizer: CountVectorizer | None = None
        self._weights: sparse.csc_matrix | None = None

    @property
    def size(self) -> int:
        return 0 if self._weights is None else self._weights.shape[0]

    def build(self, texts: Sequence[str]) -> BM25Index:
        self._vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            stop_words="english",
            dtype=np.float32,
        )
        tf = self._vectorizer.fit_transform(texts).tocsr()

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())

        n_docs = tf.shape[0]
        doc_freq = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # BM25 term weight: idf * tf * (k1 + 1) / (tf + norm[doc]).
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = (
            idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        ).astype(np.float32)

        self._weights = tf.tocsc()
        return self

    def search(
        self,
        query: str,
        k: int,
        restrict_to: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (indices, scores) of the top-k documents, best first.

        If restrict_to is given, only those document indices are ranked.
        """

        if self._weights is None or self._vectorizer is None:
            raise RuntimeError("Index has not been built.")

        terms = np.unique(self._vectorizer.transform([query]).indices)
        if len(terms) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        scores = np.asarray(self._weights[:, terms].sum(axis=1)).ravel()
        ids = np.arange(len(scores)) if restrict_to is None else np.asarray(restrict_to)
        scores = scores[ids]

        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return ids[top], scores[top]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "vectorizer": self._vectorizer,
                    "weights": self._weights,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with Path(path).open("rb") as f:
            state = pickle.load(f)

        index = cls(k1=state["k1"], b=state["b"])
        index._vectorizer = state["vectorizer"]
        index._weights = state["weights"]
        return index


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
) -> list[int]:
    """Merge ranked lists of ids; ids ranked high in any list come first."""

    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores, key=scores.__getitem__, reverse=True)
