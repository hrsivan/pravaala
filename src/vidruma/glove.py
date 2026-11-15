r"""GloVe (Pennington, Socher & Manning 2014), AdaGrad on the nonzeros of C.

Objective, over the nonzero entries of the co-occurrence matrix X:
    J = \sum_{ij: X_ij > 0} f(X_ij) ( w_i^T u_j + b_i + c_j - ln X_ij )^2,
    f(x) = min(1, (x / x_max)^beta)      (beta = 3/4, x_max = 100).

Relation to PMI: if the biases absorb the marginals, b_i ~ ln X_i and
c_j ~ ln X_j - ln N, then w_i^T u_j ~ ln(X_ij N / (X_i X_j)) = PMI(i, j).
GloVe is therefore a *weighted* least-squares factorisation of (a biased
version of) the PMI matrix, with f down-weighting rare, noisy cells; this
puts all four models of the study (SGNS, SPPMI-SVD, GloVe, FastText) in
one information-theoretic frame. Final vectors are W + U (the original
paper's choice; it averages two noisy estimates of the same quantity).

Uses the same co-occurrence matrix as the count model (embed.cooccurrence,
harmonic weighting a(d) = 1/d, as in the reference implementation).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def train_glove(C: sp.spmatrix, dim=300, epochs=25, lr=0.05, x_max=100.0,
                beta=0.75, batch=200_000, seed=0, verbose=True) -> np.ndarray:
    C = sp.coo_matrix(C)
    i, j, x = C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float64)
    keep = x > 0
    i, j, x = i[keep], j[keep], x[keep]
    logx = np.log(x)
    fx = np.minimum(1.0, (x / x_max) ** beta)
    n = C.shape[0]
    rng = np.random.default_rng(seed)
    W = (rng.random((n, dim)) - 0.5) / dim
    U = (rng.random((n, dim)) - 0.5) / dim
    b, c = np.zeros(n), np.zeros(n)
    # AdaGrad accumulators (init 1 as in the reference implementation)
    gW, gU, gb, gc = np.ones((n, dim)), np.ones((n, dim)), np.ones(n), np.ones(n)
    for ep in range(epochs):
        perm = rng.permutation(len(x))
        loss = 0.0
        for s in range(0, len(x), batch):
            k = perm[s:s + batch]
            ii, jj = i[k], j[k]
            diff = (W[ii] * U[jj]).sum(1) + b[ii] + c[jj] - logx[k]
            g = fx[k] * diff                     # dJ/d(pred) / 2
            loss += float((fx[k] * diff ** 2).sum())
            dW = g[:, None] * U[jj]
            dU = g[:, None] * W[ii]
            # accumulate per-row gradients (rows repeat within a batch)
            # scatter-add via sparse incidence matrices: DW = S_i dW, S_i[r, t] = [ii_t = r]
            ar = np.arange(len(k))
            DW = sp.csr_matrix((np.ones(len(k)), (ii, ar)), shape=(n, len(k))) @ dW
            DU = sp.csr_matrix((np.ones(len(k)), (jj, ar)), shape=(n, len(k))) @ dU
            Db = np.bincount(ii, g, n); Dc = np.bincount(jj, g, n)
            rows_w, rows_u = np.unique(ii), np.unique(jj)
            W[rows_w] -= lr * DW[rows_w] / np.sqrt(gW[rows_w])
            U[rows_u] -= lr * DU[rows_u] / np.sqrt(gU[rows_u])
            gW[rows_w] += DW[rows_w] ** 2
            gU[rows_u] += DU[rows_u] ** 2
            b -= lr * Db / np.sqrt(gb); gb += Db ** 2
            c -= lr * Dc / np.sqrt(gc); gc += Dc ** 2
        if verbose:
            print(f"glove epoch {ep + 1}/{epochs}  J/nnz = {loss / len(x):.4f}", flush=True)
    return W + U

