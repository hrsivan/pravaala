r"""Embedding construction: count-based (PPMI + truncated SVD) and predictive
(SGNS via gensim), plus a lookup layer that maps benchmark *strings* onto
view-specific *types* (needed for lemma-ID views, where one string may name
several lemmas).

Count-based model. With vocabulary V (|V| = n) and co-occurrence counts
C in R^{n x n}, let M = SPPMI(C) (see infotheory.ppmi_matrix) and its
truncated SVD M ~ U_d Sigma_d V_d^T. By Eckart-Young-Mirsky, U_d Sigma_d V_d^T
is the best rank-d approximation of M in Frobenius norm. Word vectors are
W = U_d Sigma_d^p, with p in {0, 0.5, 1} (eigenvalue weighting; Levy et al.
2015 found p = 0.5 or 0 often beats the "symmetric" p = 1).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from .dcs import strip_id
from .infotheory import ppmi_matrix


# ------------------------------------------------------------ vocab/cooc ---
def build_vocab(sentences, min_count=5):
    c = Counter()
    for s in sentences():
        c.update(s)
    words = sorted((w for w, k in c.items() if k >= min_count), key=lambda w: -c[w])
    return {w: i for i, w in enumerate(words)}, np.array([c[w] for w in words], float)


def cooccurrence(sentences, vocab, window=5, harmonic=True, keep_prob=None,
                 unweighted=False):
    r"""Symmetric co-occurrence counts within a +-window.

    C[w, c] = \sum_{occurrences} \sum_{0<|d|<=L} a(d),  a(d) = 1/|d| if
    harmonic else 1. If keep_prob (per type) is given, each pair is weighted
    by k(w) k(c): the *expected* count under word2vec-style subsampling,
    which removes the sampling noise of actually dropping tokens.
    With unweighted=True, also return R, the plain number of co-occurrences
    per cell (needed for absolute discounting).
    """
    n = len(vocab)
    C = sp.csr_matrix((n, n))
    R = sp.csr_matrix((n, n), dtype=np.float32)
    rows, cols, vals = [], [], []

    def flush():
        nonlocal C, R, rows, cols, vals
        if rows:
            r, c, v = (np.concatenate(x) for x in (rows, cols, vals))
            C = C + sp.coo_matrix((v, (r, c)), shape=(n, n)).tocsr()
            if unweighted:
                R = R + sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)).tocsr()
        rows, cols, vals = [], [], []

    for i, s in enumerate(sentences()):
        if i % 100_000 == 0:
            flush()
        ids = np.array([vocab[t] for t in s if t in vocab], dtype=np.int64)
        m = len(ids)
        for d in range(1, min(window, m - 1) + 1):
            a, b = ids[:-d], ids[d:]
            w = np.full(len(a), 1.0 / d if harmonic else 1.0)
            if keep_prob is not None:
                w *= keep_prob[a] * keep_prob[b]
            rows += [a, b]; cols += [b, a]; vals += [w, w]
    flush()
    return (C, R) if unweighted else C


def svd_embeddings(C, dim=300, alpha=0.75, shift=1.0, p=0.5, seed=0,
                   R=None, discount=0.0, ctx="alpha"):
    M = ppmi_matrix(C, alpha=alpha, shift=shift, R=R, discount=discount, ctx=ctx)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(M.shape))
    U, S, _ = svds(M, k=dim, v0=v0)
    order = np.argsort(-S)
    return U[:, order] * (S[order] ** p)


def train_sgns(corpus_file, dim=300, window=5, min_count=5, negative=5,
               sample=1e-4, epochs=5, workers=4, seed=0, fasttext=False):
    from gensim.models import FastText, Word2Vec
    cls = FastText if fasttext else Word2Vec
    m = cls(corpus_file=str(corpus_file), vector_size=dim, window=window,
            min_count=min_count, sg=1, negative=negative, sample=sample,
            epochs=epochs, workers=workers, seed=seed)
    return m.wv


# --------------------------------------------------------------- lookup -----
@dataclass
class Space:
    """A unit-normalised embedding space with string-level lookup.

    For ID views, `senses[s]` lists the types t with strip(t) = s, together
    with their corpus counts. `vec(s, how)` resolves the string:
        'maxfreq' : vector of the most frequent type
        'mean'    : count-weighted mean of type vectors (renormalised)
    and `sim(s, t, how='maxsim')` = max_{a in s, b in t} cos(a, b)
    (multi-prototype MaxSim; Reisinger & Mooney 2010).
    """
    words: list[str]
    X: np.ndarray            # (n, d), rows unit-norm
    counts: np.ndarray

    def __post_init__(self):
        self.X = self.X / np.maximum(np.linalg.norm(self.X, axis=1, keepdims=True), 1e-12)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.senses = defaultdict(list)
        for i, w in enumerate(self.words):
            self.senses[strip_id(w)].append(i)

    def save(self, path):
        np.savez_compressed(path, words=np.array(self.words), X=self.X.astype(np.float32),
                            counts=self.counts)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["words"].tolist(), z["X"].astype(np.float64), z["counts"])

    @classmethod
    def from_kv(cls, kv):
        return cls(list(kv.index_to_key), np.asarray(kv.vectors),
                   np.array([kv.get_vecattr(w, "count") for w in kv.index_to_key], float))

    # Optional string normaliser applied to every *lookup* (benchmark word),
    # set when the corpus was built with an orthographic profile (e.g. nasal
    # canonicalisation), so benchmark words get exactly the same treatment.
    canon = None
    # Optional spelling-variant fallback for OOV lookups (normalise.Respeller);
    # enable with use_respelling(). Off by default.
    respell = None

    def use_respelling(self):
        from .normalise import Respeller
        self.respell = Respeller({k: float(self.counts[ids].sum()) for k, ids in self.senses.items()
                                  if ids})
        return self

    def _k(self, s):
        k = self.canon(s) if self.canon else s
        if self.respell is not None and k not in self.senses:
            k = self.respell(k) or k
        return k

    def __contains__(self, s):
        return self._k(s) in self.senses

    def vec(self, s, how="maxfreq"):
        ids = self.senses[self._k(s)]
        if how == "maxfreq":
            return self.X[max(ids, key=lambda i: self.counts[i])]
        v = (self.counts[ids, None] * self.X[ids]).sum(0)
        return v / np.linalg.norm(v)

    def sim(self, s, t, how="maxsim"):
        if how == "maxsim":
            return float((self.X[self.senses[self._k(s)]] @ self.X[self.senses[self._k(t)]].T).max())
        return float(self.vec(s, how) @ self.vec(t, how))
