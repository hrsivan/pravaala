import numpy as np
import scipy.sparse as sp

from vidruma.evaluate import auc, purity
from vidruma.infotheory import (conditional_entropy, entropy, mutual_information,
                                nmi, ppmi_matrix, surprisal_keep_prob,
                                variation_of_information)


def test_entropy_uniform():
    assert np.isclose(entropy([1, 1, 1, 1]), 2.0)


def test_chain_rule():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 500).tolist()
    y = [(a + rng.integers(0, 2)) % 5 for a in x]
    hxy = entropy(np.unique(np.c_[x, y], axis=0, return_counts=True)[1])
    assert np.isclose(hxy, entropy(np.bincount(x)) + conditional_entropy(x, y))


def test_function_has_zero_conditional_entropy():
    x = list(range(10)) * 3
    assert np.isclose(conditional_entropy(x, [v % 3 for v in x]), 0.0)


def test_vi_metric_axioms():
    rng = np.random.default_rng(1)
    a, b, c = (rng.integers(0, 5, 200).tolist() for _ in range(3))
    assert np.isclose(variation_of_information(a, a), 0.0)
    assert np.isclose(variation_of_information(a, b), variation_of_information(b, a))
    assert (variation_of_information(a, c)
            <= variation_of_information(a, b) + variation_of_information(b, c) + 1e-12)


def test_nmi_bounds():
    x = [0, 0, 1, 1, 2, 2]
    assert np.isclose(nmi(x, [5, 5, 7, 7, 9, 9]), 1.0)
    assert 0 <= mutual_information(x, [0, 1, 0, 1, 0, 1]) <= 1e-12


def test_purity_monotone_under_refinement():
    gold = [0, 0, 1, 1, 1, 2, 2, 0]
    coarse = [0, 0, 0, 0, 1, 1, 1, 1]
    fine = [0, 0, 2, 2, 1, 1, 3, 3]          # refines `coarse`
    assert purity(fine, gold) >= purity(coarse, gold)
    assert purity(list(range(8)), gold) == 1.0


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 2], [1, 2]) == 0.5


def test_ppmi_independent_is_zero():
    # rank-one counts = independence => PMI = 0 everywhere (alpha=1)
    C = sp.csr_matrix(np.outer([1, 2, 3], [2, 2, 4]).astype(float))
    assert ppmi_matrix(C, alpha=1.0).nnz == 0


def test_subsampling_matches_word2vec():
    counts = np.array([1e6, 1e3, 10.0])
    f = counts / counts.sum()
    assert np.allclose(surprisal_keep_prob(counts, 1e-4),
                       np.minimum(1, np.sqrt(1e-4 / f)))


def test_discount_is_count_dependent_shift():
    import numpy as np, scipy.sparse as sp
    from vidruma.infotheory import ppmi_matrix, ney_discount
    R = sp.csr_matrix(np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]], float))
    C = R * 0.5                                   # weighted counts, same support
    P0 = ppmi_matrix(C, alpha=1.0).toarray()
    P1 = ppmi_matrix(C, alpha=1.0, R=R, discount=0.5).toarray()
    full = np.log(C.toarray() / C.sum() / np.outer(C.sum(1).A1, C.sum(0).A1) * C.sum() ** 2,
                  where=C.toarray() > 0, out=np.full((3, 3), -np.inf))
    shift = np.log(np.maximum(1 - 0.5 / np.where(R.toarray() > 0, R.toarray(), 1), 1e-300))
    expect = np.maximum(np.where(R.toarray() > 0, full + shift, 0), 0)
    assert np.allclose(P1, expect) and np.allclose(P0, np.maximum(np.where(R.toarray() > 0, full, 0), 0))
    assert ney_discount(R) == 2 / (2 + 2 * 2)      # n1 = 2, n2 = 2


def test_continuation_marginal_counts_partners():
    import numpy as np, scipy.sparse as sp
    from vidruma.infotheory import ppmi_matrix
    # context 0: 3 distinct partners once each; context 1: one partner 3 times
    C = sp.csr_matrix(np.array([[1, 3], [1, 0], [1, 0]], float))
    M = ppmi_matrix(C, ctx="cont").toarray()
    # Q = (3/4, 1/4); P(w0)=4/6; PMI(w0,c1) = log((3/6)/((4/6)(1/4))) = log 3
    assert np.isclose(M[0, 1], np.log(3))
