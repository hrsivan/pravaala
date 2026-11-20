from collections import Counter

import numpy as np
import scipy.sparse as sp

from vidruma.embed import Space
from vidruma.plugins import compose


def test_segmenter_prefers_low_surprisal_split():
    lex = Counter({"jala": 500, "ja": 5, "laprAya": 5, "prAya": 300, "padma": 200,
                   "ākara": 5, "Akara": 100})
    seg = compose.Segmenter(lex, min_count=5)
    best = seg.kbest("jalaprAya", 2)
    assert best[0][1] == ("jala", "prAya")
    # vowel sandhi: padma + Akara -> padmAkara
    assert seg.kbest("padmAkara", 1)[0][1] == ("padma", "Akara")


def test_candidates_flags_particles():
    seg = compose.Segmenter(Counter({"akzi": 100, "iva": 1000}), min_count=5)
    row = compose.candidates({"akzIva"}, seg)[0]
    assert row["split"] == "akzi+iva" and row["status"] == "needs_review"


def test_augment_does_not_modify_original():
    X = np.eye(3)
    s = Space(["a", "b", "c"], X.copy(), np.array([5.0, 5.0, 5.0]))
    aug, added = compose.augment(s, {"ab": ["a", "b"], "zz": ["a", "q"]})
    assert added == ["ab"] and "ab" not in s and "ab" in aug
    assert np.isclose(aug.vec("ab") @ aug.vec("a"), 1 / np.sqrt(2))


def test_space_save_load(tmp_path):
    s = Space(["x_1", "x_2", "y_3"], np.random.default_rng(0).normal(size=(3, 4)),
              np.array([3.0, 7.0, 1.0]))
    s.save(tmp_path / "v.npz")
    t = Space.load(tmp_path / "v.npz")
    assert t.words == s.words and np.allclose(t.X, s.X, atol=1e-6)
    assert set(t.senses["x"]) == {0, 1} and t.vec("x", "maxfreq") @ t.X[1] > 0.999
