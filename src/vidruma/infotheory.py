r"""Information-theoretic quantities used throughout the project.

Conventions. Logarithms are base 2 (bits) unless `base` is given. For a
finite random variable X with pmf p, H(X) = -\sum_x p(x) log p(x), with
0 log 0 := 0. Plug-in (maximum-likelihood) estimators are used throughout;
`miller_madow` adds the first-order bias correction (m-1)/(2N ln 2), where m
is the number of observed outcomes, since the plug-in entropy estimator is
negatively biased for small samples.

Contents
  entropy, conditional_entropy, mutual_information    (from counts / pairs)
  nmi, variation_of_information                       (partition comparison)
  good_turing_unseen                                  (unseen-type mass)
  ppmi_matrix                                         (PMI weighting)
  surprisal_keep_prob                                 (frequency damping)
"""
from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------- entropies ---
def entropy(counts: Iterable[float], base: float = 2.0,
            miller_madow: bool = False) -> float:
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    n = c.sum()
    if n == 0:
        return 0.0
    p = c / n
    h = float(-(p * np.log(p)).sum())
    if miller_madow:
        h += (len(c) - 1) / (2 * n)
    return h / np.log(base)


def joint_counts(xs: Sequence[Hashable], ys: Sequence[Hashable]) -> Counter:
    assert len(xs) == len(ys)
    return Counter(zip(xs, ys))


def conditional_entropy(xs, ys, base=2.0) -> float:
    """H(Y | X) = H(X, Y) - H(X), from paired observations."""
    j = joint_counts(xs, ys)
    return entropy(j.values(), base) - entropy(Counter(xs).values(), base)


def mutual_information(xs, ys, base=2.0) -> float:
    """I(X; Y) = H(X) + H(Y) - H(X, Y)."""
    return (entropy(Counter(xs).values(), base) + entropy(Counter(ys).values(), base)
            - entropy(joint_counts(xs, ys).values(), base))


def nmi(xs, ys) -> float:
    """Normalised MI, arithmetic normalisation: 2 I(X;Y) / (H(X) + H(Y))."""
    hx, hy = entropy(Counter(xs).values()), entropy(Counter(ys).values())
    return 0.0 if hx + hy == 0 else 2 * mutual_information(xs, ys) / (hx + hy)


def variation_of_information(xs, ys) -> float:
    """VI(X, Y) = H(X|Y) + H(Y|X) = 2 H(X,Y) - H(X) - H(Y).

    Meila (2007): VI is a metric on the lattice of partitions of a finite
    set, bounded by log2 n. Unlike purity it is not monotone in the number
    of clusters, which is why we report it alongside purity.
    """
    hxy = entropy(joint_counts(xs, ys).values())
    return 2 * hxy - entropy(Counter(xs).values()) - entropy(Counter(ys).values())


# ------------------------------------------------- unseen mass / sparsity ---
def good_turing_unseen(token_counts: Counter) -> float:
    """Good-Turing estimate of P(next token is an unseen type) = N_1 / N.

    Under each corpus view this is a principled, model-free estimate of the
    OOV rate a *random running token* would face; comparing it across views
    measures how much sparsity each normalisation removes (RQ3).
    """
    n = sum(token_counts.values())
    n1 = sum(1 for c in token_counts.values() if c == 1)
    return n1 / n if n else 0.0


# --------------------------------------------------------------- PMI ---------
def ney_discount(R: sp.spmatrix) -> float:
    r"""Ney, Essen & Kneser (1994) absolute discount D = n_1 / (n_1 + 2 n_2),
    where n_r is the number of cells of the (unweighted) count matrix R seen
    exactly r times."""
    d = sp.csr_matrix(R).data
    n1, n2 = np.sum(np.isclose(d, 1)), np.sum(np.isclose(d, 2))
    return float(n1 / (n1 + 2 * n2))


def ppmi_matrix(C: sp.spmatrix, alpha: float = 0.75, shift: float = 1.0,
                R: sp.spmatrix | None = None, discount: float | str = 0.0,
                ctx: str = "alpha") -> sp.csr_matrix:
    r"""Shifted positive PMI with optional smoothing (natural log).

        PMI(w, c)   = log  P(w, c) / ( P(w) Q(c) )
        SPPMI(w, c) = max(PMI(w, c) - log shift, 0)

    Context marginal Q (``ctx``):
      "alpha": Q(c) \propto #(c)^alpha, context-distribution smoothing
               (Levy, Goldberg & Dagan 2015); alpha=1 is no smoothing;
      "cont":  Q(c) \propto |{w : #(w,c) > 0}|, the Kneser-Ney continuation
               probability: how many distinct words c occurs with.

    Absolute discounting (``discount`` = D, or "ney" for D estimated from
    count-of-counts; needs the unweighted count matrix R): each cell's
    weighted count is scaled by max(r - D, 0) / r, r its unweighted count,
    while the marginals keep the raw counts. Hence
        PMI_D(w, c) = PMI(w, c) + log(1 - D / r(w, c)),
    a count-dependent shift: log 4 for a pair seen once, ~0 for frequent ones.

    SGNS with k negative samples implicitly factorises PMI - log k (Levy &
    Goldberg 2014): the same shift for every cell.
    """
    C = sp.csr_matrix(C, dtype=np.float64)
    total = C.sum()
    pw = np.asarray(C.sum(axis=1)).ravel() / total
    if ctx == "alpha":
        cc = np.asarray(C.sum(axis=0)).ravel() ** alpha
    elif ctx == "cont":
        cc = np.diff(sp.csc_matrix(C).indptr).astype(float)
    else:
        raise ValueError(ctx)
    pc = cc / cc.sum()
    C = C.tocoo()
    pmi = np.log(C.data / total) - np.log(pw[C.row]) - np.log(pc[C.col])
    if discount == "ney" or (discount and discount > 0):
        if R is None:
            raise ValueError("discounting needs the unweighted count matrix R")
        D = ney_discount(R) if discount == "ney" else float(discount)
        r = np.asarray(sp.csr_matrix(R)[C.row, C.col]).ravel()
        with np.errstate(divide="ignore"):
            pmi = pmi + np.log(np.maximum(1.0 - D / r, 0.0))
    pmi -= np.log(shift)
    keep = pmi > 1e-12          # tolerance: exact independence gives 0 up to rounding
    return sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)


# --------------------------------------------------- frequency damping -----
def surprisal_keep_prob(counts: np.ndarray, t: float = 1e-4) -> np.ndarray:
    r"""Mikolov et al. (2013) subsampling, rewritten in terms of surprisal.

    word2vec keeps an occurrence of w with probability
        k(w) = min(1, sqrt(t / f(w))),  f(w) = #(w)/N.
    With self-information  I(w) = -log2 f(w)  and  I_t = -log2 t  this is
        k(w) = min(1, 2^{-(I_t - I(w))/2}),
    i.e. occurrences carrying less information than the threshold I_t bits
    are discarded at a rate exponential in their information deficit. This
    form makes the threshold interpretable (bits) and lets us compare it to
    alternatives (e.g. weighting by I(w) or by context entropy).
    """
    f = counts / counts.sum()
    surprisal = -np.log2(np.maximum(f, 1e-300))
    return np.minimum(1.0, 2.0 ** (-(-np.log2(t) - surprisal) / 2))


def context_entropy(C: sp.csr_matrix, base: float = 2.0) -> np.ndarray:
    """Row-wise H(C | W = w): entropy of each word's context distribution.

    Function words have near-maximal context entropy (they co-occur with
    everything); we test whether H(C|w) differs systematically between
    concrete and abstract vargas (RQ2) and use it as an alternative
    frequency-independent damping weight.
    """
    C = sp.csr_matrix(C, dtype=float)
    out = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        row = C.data[C.indptr[i]:C.indptr[i + 1]]
        out[i] = entropy(row, base)
    return out
