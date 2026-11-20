r"""Intrinsic evaluation with explicit coverage and uncertainty.

Design principles (following Schnabel et al. 2015; Faruqui et al. 2016):
  * Every score is reported together with its coverage |B_cov| / |B|; OOV
    items are *excluded*, never imputed (EvalSan imputes the mean vector,
    which assigns cos = 1 to any pair of two OOV words).
  * Cosine is used only through rank statistics (Spearman rho, AUC), which
    are invariant under monotone transformations of the similarity scale and
    hence comparable across embedding spaces with different cosine
    distributions (anisotropy).
  * Accuracies carry Wilson score intervals; cluster scores carry a
    permutation null (the same cluster sizes, gold labels shuffled).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans

from .infotheory import nmi, variation_of_information


def wilson(k, n, conf=0.95):
    if n == 0:
        return (float("nan"),) * 2
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def auc(pos, neg):
    """P(S_pos > S_neg) + 0.5 P(=): the Mann-Whitney U statistic / (|P||N|)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r = rankdata(allv)[: len(pos)]
    return (r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ----------------------------------------------------------- relatedness ---
def relatedness(space, pairs, how="maxsim"):
    cov = [(a, b, y) for a, b, y in pairs if a in space and b in space]
    s = np.array([space.sim(a, b, how) for a, b, _ in cov])
    y = np.array([y for *_, y in cov])
    res = {"coverage": len(cov) / len(pairs), "n": len(cov)}
    if len(cov) > 2:
        res["spearman"] = float(spearmanr(s, y).statistic)
        for hi, lo in [(2, 0), (1, 0), (2, 1)]:
            res[f"auc_{hi}v{lo}"] = float(auc(s[y == hi], s[y == lo]))
    return res, (s, y)


def tune_thresholds(s, y):
    """Grid-search (t1 < t2) maximising macro-F1 of the 3-way rule on dev."""
    from sklearn.metrics import f1_score
    grid = np.quantile(s, np.linspace(0.02, 0.98, 49))
    best = (-1, None)
    for t1, t2 in itertools.combinations(grid, 2):
        f = f1_score(y, np.digitize(s, [t1, t2]), average="macro")
        best = max(best, (f, (t1, t2)), key=lambda z: z[0])
    return best


def classify_relatedness(s, y, thresholds):
    from sklearn.metrics import f1_score
    return float(f1_score(y, np.digitize(s, thresholds), average="macro"))


# -------------------------------------------------------------- synonyms ---
def synonym_mcq(space, items, how="maxsim"):
    cov = [(q, o) for q, o in items if q in space and all(x in space for x in o)]
    k = sum(int(np.argmax([space.sim(q, x, how) for x in o]) == 0) for q, o in cov)
    n = len(cov)
    return {"coverage": n / len(items), "n": n, "acc": k / n if n else float("nan"),
            "wilson95": wilson(k, n), "chance": 0.25}


# ------------------------------------------------------------ clustering ---
def purity(clusters, gold):
    """(1/N) sum_k max_j |c_k  cap  g_j|. Monotone non-decreasing in #clusters."""
    clusters, gold = np.asarray(clusters), np.asarray(gold)
    tot = 0
    for k in np.unique(clusters):
        _, cnt = np.unique(gold[clusters == k], return_counts=True)
        tot += cnt.max()
    return tot / len(gold)


def cluster_scores(X, gold, k=None, method="kmeans", n_perm=200, seed=0):
    """Cluster unit-norm rows of X into k groups and compare with gold labels.

    k-means on unit vectors approximates spherical k-means (cosine geometry):
    for unit x, mu,  ||x - mu||^2 = 2 - 2 cos(x, mu) when ||mu|| = 1.
    Returns purity, NMI, VI and their permutation-null means/p-values.
    """
    gold = np.asarray(gold)
    k = k or len(set(gold))
    if method == "kmeans":
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    elif method == "ward":
        # Ward on unit vectors: ||x-y||^2 = 2 - 2cos(x,y), so the Euclidean
        # geometry Ward needs is a monotone function of cosine distance.
        lab = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    else:  # average linkage; tends to chain into one giant cluster
        lab = AgglomerativeClustering(k, metric="cosine", linkage="average").fit_predict(X)
    obs = {"purity": purity(lab, gold), "nmi": nmi(lab, gold),
           "vi": variation_of_information(lab, gold)}
    rng = np.random.default_rng(seed)
    null = {m: [] for m in obs}
    for _ in range(n_perm):
        g = rng.permutation(gold)
        null["purity"].append(purity(lab, g))
        null["nmi"].append(nmi(lab, g))
        null["vi"].append(variation_of_information(lab, g))
    out = {"k": k, "n": len(gold)}
    for m, v in obs.items():
        nv = np.array(null[m])
        better = (nv <= v) if m == "vi" else (nv >= v)   # VI: lower is better
        out[m] = float(v)
        out[f"{m}_null"] = float(nv.mean())
        out[f"{m}_p"] = float((better.sum() + 1) / (n_perm + 1))
    return out, lab
