r"""GloVe (Pennington, Socher & Manning 2014), via the reference implementation.

Objective, over the nonzero entries of the co-occurrence matrix X:
    J = \sum_{ij: X_ij > 0} f(X_ij) ( w_i^T u_j + b_i + c_j - ln X_ij )^2,
    f(x) = min(1, (x / x_max)^beta).

Relation to PMI: if the biases absorb the marginals, b_i ~ ln X_i and
c_j ~ ln X_j - ln N, then w_i^T u_j ~ ln(X_ij N / (X_i X_j)) = PMI(i, j).
GloVe is therefore a *weighted* least-squares factorisation of (a biased
version of) the PMI matrix, which puts all four models of the study
(word2vec/SGNS, PPMI-SVD, GloVe, FastText) in one information-theoretic frame.

An earlier AdaGrad implementation of our own stayed near chance on the
benchmarks and was replaced by the reference code (github.com/stanfordnlp/GloVe).
"""
from __future__ import annotations

import numpy as np


def train_glove_reference(corpus, bin_dir, dim=100, window=5, min_count=5, epochs=50,
                          x_max=10.0, alpha=0.75, eta=0.05, threads=2, seed=0,
                          workdir=None, memory_gb=2.0):
    r"""GloVe with the reference implementation (Pennington et al. 2014;
    github.com/stanfordnlp/GloVe, built with `make`; `bin_dir` is its build/).

    Same corpus file and window as the other models; symmetric window with the
    reference's 1/d distance weighting, as in our own co-occurrence counts.
    Vectors are W + C (`-model 2`, as in the paper). x_max = 10 follows the
    reference demo for small corpora (the paper's 100 was set for billions of
    tokens). Returns (words, X, counts) in vocabulary order.
    """
    import subprocess
    import tempfile
    from pathlib import Path
    bin_dir = Path(bin_dir)
    wd = Path(workdir or tempfile.mkdtemp(prefix="glove_"))
    wd.mkdir(parents=True, exist_ok=True)

    def run(cmd, stdin, stdout):
        with open(stdin, "rb") as fi, open(stdout, "wb") as fo:
            subprocess.run(cmd, stdin=fi, stdout=fo, check=True, stderr=subprocess.DEVNULL)

    run([bin_dir / "vocab_count", "-min-count", str(min_count), "-verbose", "0"],
        corpus, wd / "vocab.txt")
    run([bin_dir / "cooccur", "-memory", str(memory_gb), "-vocab-file", wd / "vocab.txt",
         "-verbose", "0", "-window-size", str(window), "-symmetric", "1"], corpus, wd / "cooc.bin")
    run([bin_dir / "shuffle", "-memory", str(memory_gb), "-verbose", "0"],
        wd / "cooc.bin", wd / "cooc.shuf.bin")
    subprocess.run([bin_dir / "glove", "-save-file", wd / "vectors", "-threads", str(threads),
                    "-input-file", wd / "cooc.shuf.bin", "-x-max", str(x_max), "-alpha", str(alpha),
                    "-eta", str(eta), "-iter", str(epochs), "-vector-size", str(dim),
                    "-binary", "0", "-model", "2", "-vocab-file", wd / "vocab.txt",
                    "-verbose", "0", "-seed", str(seed)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    counts = {}
    for line in open(wd / "vocab.txt", encoding="utf-8"):
        w, c = line.rstrip("\n").rsplit(" ", 1)
        counts[w] = int(c)
    words, rows = [], []
    for line in open(wd / "vectors.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split(" ")
        if parts[0] in counts:                   # skips the reference's <unk> row
            words.append(parts[0]); rows.append(np.asarray(parts[1:], dtype=np.float32))
    return words, np.stack(rows), np.array([counts[w] for w in words], dtype=float)
