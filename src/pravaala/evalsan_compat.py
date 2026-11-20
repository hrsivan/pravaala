"""EvalSan-faithful evaluation protocol (for comparability with Sandhan et al.).

Self-contained: nothing else in the package imports this module, so it can
be removed (together with the "evalsan" protocol key in result files)
without affecting the main protocol in evaluate.py.

Replicates, task by task, the behaviour of EvalSan's main.py / web/evaluate.py:
  relatedness    : OOV words -> mean vector; fixed thresholds (0.25, 0.5) for
                   static models; accuracy and macro-F1 on the full test set.
  synonym MCQ    : items with any OOV word are *skipped*; accuracy.
  categorisation : OOV words -> mean vector; purity maximised over
                   {Ward, average/complete x cosine/euclidean, k-means}
                   (selection on the test set).
The one deliberate difference: benchmark words are transliteration-normalised
before lookup unless scheme="raw" is passed to the loaders, which reproduces
the original WX/SLP1 mismatch exactly.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, f1_score

from .evaluate import purity


def _vec_or_mean(space, w, mean, how):
    return space.vec(w, how) if w in space else mean


def relatedness(space, pairs, how="maxfreq", thresholds=(0.25, 0.5)):
    mean = space.X.mean(0)
    A = np.stack([_vec_or_mean(space, a, mean, how) for a, _, _ in pairs])
    B = np.stack([_vec_or_mean(space, b, mean, how) for _, b, _ in pairs])
    s = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
    y = np.array([y for *_, y in pairs])
    pred = np.digitize(s, thresholds)
    n_oov_pairs = sum(not (a in space and b in space) for a, b, _ in pairs)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macroF1": float(f1_score(y, pred, average="macro")),
            "n": len(pairs), "pairs_with_oov": n_oov_pairs,
            "both_oov_pairs": sum(a not in space and b not in space for a, b, _ in pairs)}


def synonym_mcq(space, items, how="maxfreq"):
    cov = [(q, o) for q, o in items if q in space and all(x in space for x in o)]
    k = sum(int(np.argmax([space.vec(q, how) @ space.vec(x, how) for x in o]) == 0)
            for q, o in cov)
    return {"accuracy": k / len(cov) if cov else float("nan"), "n": len(cov)}


def categorisation(space, cats: dict, how="maxfreq", seed=0):
    mean = space.X.mean(0)
    words = [w for ws in cats.values() for w in ws]
    y = np.array([c for c, ws in cats.items() for _ in ws])
    X = np.stack([_vec_or_mean(space, w, mean, how) for w in words])
    k = len(cats)
    ids = np.random.RandomState(seed).permutation(len(words))
    X, y = X[ids], y[ids]
    best = purity(AgglomerativeClustering(k, linkage="ward").fit_predict(X), y)
    for metric in ("cosine", "euclidean"):
        for link in ("average", "complete"):
            lab = AgglomerativeClustering(k, metric=metric, linkage=link).fit_predict(X)
            best = max(best, purity(lab, y))
    best = max(best, purity(KMeans(k, n_init=10, random_state=seed).fit_predict(X), y))
    return {"purity_best_of_6": float(best), "n": len(words),
            "oov": sum(w not in space for w in words)}
