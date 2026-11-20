r"""OPTIONAL PLUGIN: compositional vectors for words absent from the corpus.

Nothing in the core package imports this module. Deleting `plugins/` (and
the "compose" key in result files) removes it without trace.

Pipeline
  1. candidates(): for each benchmark word absent from the lemma vocabulary,
     propose a segmentation into attested lemmas, undoing the commonest
     junction sandhi and stem alternations.
  2. A human reviews the TSV (status column: auto / needs_review /
     participle_review  ->  approved / rejected; splits may be edited).
     By default ONLY rows marked `approved` are used: `auto` is a sorting
     hint (high-margin, no warning flags), not an acceptance, because a
     low-surprisal split can still be a false etymology (akzIva "drunk" is
     not akzi+iva). `--compose_accept approved,auto` opts in explicitly.
  3. augment(): add a vector for every accepted word,
         v(w) = (sum_i v(m_i)) / || sum_i v(m_i) ||     (additive composition,
     Mitchell & Lapata 2010). Additive composition is order-blind: it cannot
     distinguish a tatpurusa (head-final) from a dvandva; treat its results
     as a lower bound on what composition can do.

Segmentation as minimum description length. With a unigram lemma model
p(m) = #(m)/N, choose the segmentation s = (m_1..m_r) of w minimising the
total surprisal  L(s) = sum_i -log2 p(m_i)  (bits). Extra members cost extra
bits, so over-segmentation is penalised without an ad-hoc length prior.
Confidence is the margin  Delta = L(s_2) - L(s_1)  between the best and the
second-best segmentation: by Bayes with a uniform prior over segmentations,
the posterior odds of s_1 over s_2 are 2^Delta.
"""
from __future__ import annotations

import csv
import heapq
import math
from collections import Counter
from functools import lru_cache

import numpy as np

UPASARGA = {"pra", "parA", "apa", "sam", "saM", "anu", "ava", "nis", "nir", "niz",
            "dus", "dur", "duz", "vi", "A", "ni", "aDi", "api", "ati", "su", "ud",
            "ut", "aBi", "prati", "pari", "upa"}

PARTICLES = {"iva", "ca", "api", "eva", "tu", "hi", "vA", "na", "ha", "sma", "iti"}

# Junction sandhi: surface string at the junction -> (left ending, right start)
_VOWEL = {
    "A": [("a", "a"), ("a", "A"), ("A", "a"), ("A", "A")],
    "I": [("i", "i"), ("i", "I"), ("I", "i"), ("I", "I")],
    "U": [("u", "u"), ("u", "U"), ("U", "u"), ("U", "U")],
    "e": [("a", "i"), ("a", "I"), ("A", "i"), ("A", "I")],
    "o": [("a", "u"), ("a", "U"), ("A", "u"), ("A", "U"), ("as", "")],
    "E": [("a", "e"), ("a", "E"), ("A", "e")],
    "O": [("a", "o"), ("a", "O"), ("A", "o")],
}
# Stem alternations of a compound's prior member: surface ending -> lemma ending
_STEM = {"a": ["an", "at"], "A": ["at", "an"], "i": ["in"], "d": ["t"], "g": ["k", "c"],
         "q": ["w"], "b": ["p"], "r": ["s"], "M": ["m", "n"], "N": ["n"]}


def _junctions(w: str, i: int):
    """Underlying (left, right) pairs for a split of surface w before index i."""
    L, R = w[:i], w[i:]
    yield L, R
    for e in _STEM.get(L[-1:], []):
        yield L[:-1] + e, R
    if L and L[-1] in _VOWEL:                     # coalesced vowel at L[-1]
        for le, rs in _VOWEL[L[-1]]:
            yield L[:-1] + le, rs + R
    if R[:1] in ("y", "v") and len(R) > 1 and R[1] in "aAuUeEoO":
        yield L + ("i" if R[0] == "y" else "u"), R[1:]


class Segmenter:
    def __init__(self, lemma_counts: Counter, min_count=5, min_len=2, max_members=4):
        self.lex = {m: c for m, c in lemma_counts.items()
                    if c >= min_count and len(m) >= min_len}
        self.N = sum(self.lex.values())
        self.max_members = max_members

    def bits(self, m):
        c = self.lex.get(m)
        return None if c is None else -math.log2(c / self.N)

    def kbest(self, w: str, k=2):
        """k lowest-surprisal segmentations of w (each >= 2 members)."""
        @lru_cache(None)
        def segs(s: str, depth: int):
            out = []
            b = self.bits(s)
            if b is not None:
                out.append((b, (s,)))
            if depth < self.max_members:
                for i in range(2, len(s) - 1):
                    for left, right in _junctions(s, i):
                        bl = self.bits(left)
                        if bl is None or len(right) < 2:
                            continue
                        for br, rest in segs(right, depth + 1):
                            out.append((bl + br, (left, *rest)))
            return tuple(heapq.nsmallest(k + 3, set(out)))
        res = [s for s in segs(w, 1) if len(s[1]) >= 2]
        return heapq.nsmallest(k, res)


def candidates(absent_words, seg: Segmenter, margin_bits=4.0):
    """Rows for the human-review TSV."""
    rows = []
    for w in sorted(absent_words):
        best = seg.kbest(w, 2)
        if not best:
            status = "participle_review" if w.endswith(("ta", "na", "ita", "Ta", "qa", "Da")) else "no_split"
            rows.append({"word": w, "split": "", "bits": "", "margin_bits": "",
                         "status": status, "note": ""})
            continue
        (b1, s1), rest = best[0], best[1:]
        margin = (rest[0][0] - b1) if rest else float("inf")
        note = []
        if any(m in UPASARGA for m in s1):
            note.append("upasarga: prefixed derivative, not a compound?")
        if w.endswith(("ta", "na", "ita")):
            note.append("participle-like ending")
        if any(m in PARTICLES for m in s1):
            note.append("particle as member")
        if any(len(m) < 3 for m in s1):
            note.append("very short member")
        status = "auto" if margin >= margin_bits and not note else "needs_review"
        rows.append({"word": w, "split": "+".join(s1), "bits": f"{b1:.2f}",
                     "margin_bits": f"{margin:.2f}", "status": status,
                     "note": "; ".join(note)})
    return rows


FIELDS = ["word", "split", "bits", "margin_bits", "status", "note"]


def write_tsv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def read_accepted(path, accept=("approved",)):
    with open(path, encoding="utf-8") as fh:
        return {r["word"]: r["split"].split("+") for r in csv.DictReader(fh, delimiter="\t")
                if r["status"] in accept and r["split"]}


def augment(space, compositions: dict[str, list[str]]):
    """Return a new Space with one composed vector per accepted word whose
    members are all in-vocabulary. The original space is not modified."""
    from ..embed import Space
    words, vecs, counts = [], [], []
    for w, members in compositions.items():
        if w in space or not all(m in space for m in members):
            continue
        v = np.sum([space.vec(m, "maxfreq") for m in members], axis=0)
        words.append(w)
        vecs.append(v / np.linalg.norm(v))
        counts.append(min(space.counts[space.senses[m]].max() for m in members))
    if not words:
        return space, []
    return Space(list(space.words) + words, np.vstack([space.X, np.array(vecs)]),
                 np.concatenate([space.counts, counts])), words
