from collections import Counter

import numpy as np
import scipy.sparse as sp

from vidruma.embed import Space
from vidruma.glove import train_glove


def test_glove_loss_decreases_and_shapes():
    rng = np.random.default_rng(0)
    C = sp.random(40, 40, density=0.3, random_state=0, data_rvs=lambda n: rng.integers(1, 50, n))
    C = C + C.T
    W = train_glove(C, dim=8, epochs=2, verbose=False)
    assert W.shape == (40, 8) and np.isfinite(W).all()


def test_space_save_load(tmp_path):
    s = Space(["x_1", "x_2", "y_3"], np.random.default_rng(0).normal(size=(3, 4)),
              np.array([3.0, 7.0, 1.0]))
    s.save(tmp_path / "v.npz")
    t = Space.load(tmp_path / "v.npz")
    assert t.words == s.words and np.allclose(t.X, s.X, atol=1e-6)
    assert set(t.senses["x"]) == {0, 1} and t.vec("x", "maxfreq") @ t.X[1] > 0.999
