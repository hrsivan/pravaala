"""Transliteration between IAST, Devanagari, WX and SLP1.

SLP1 is the canonical internal alphabet of the project: it is a bijection
between Sanskrit phonemes and single ASCII characters, so string length equals
phoneme count, and character n-gram statistics are statistics over phonemes.

Formally, let Sigma be the Sanskrit phoneme inventory. Each scheme S defines an
injective encoding  enc_S : Sigma -> Gamma_S^+  (Gamma_S a character set). SLP1
and WX are *letter-to-letter* (|enc(sigma)| = 1 for all sigma), so the map
WX -> SLP1 is a bijection between the two letter sets and exactly invertible
(the only exception is SLP1 'X', long vocalic l, which WX cannot write as one
letter; it is a grammarians' phoneme and does not occur in our data). IAST is
not prefix-free ("a" is a prefix of "ai"), so decoding uses greedy longest
match; the rare true hiatus a+i is written "aï" in IAST and handled explicitly.
"""
from __future__ import annotations

import unicodedata

__all__ = ["iast_to_slp1", "wx_to_slp1", "slp1_to_wx", "deva_to_slp1",
           "slp1_to_iast", "detect_scheme"]

# --------------------------------------------------------------------------
# WX <-> SLP1: one character permutation (both are letter-to-letter schemes)
# --------------------------------------------------------------------------
# Only letters whose code differs are listed; a, A, i, I, u, U, e, E, o, O,
# k, K, g, G, c, C, j, J, p, P, b, B, m, y, r, l, v, s, h, M, H are shared.
_WX2SLP1 = {
    "t": "w", "T": "W", "d": "q", "D": "Q", "N": "R",   # retroflex stops, nasal
    "w": "t", "W": "T", "x": "d", "X": "D",              # dental stops
    "f": "N", "F": "Y",                                  # velar / palatal nasal
    "R": "z",                                            # retroflex sibilant
    "q": "f", "Q": "F", "L": "x",                        # vocalic r, rr, l
    "z": "~",                                            # candrabindu
}
_SLP12WX = {v: k for k, v in _WX2SLP1.items()}
assert len(_SLP12WX) == len(_WX2SLP1), "WX<->SLP1 map must be a bijection"
_WX_TABLE = str.maketrans(_WX2SLP1)
_SLP1_WX_TABLE = str.maketrans(_SLP12WX)


def wx_to_slp1(s: str) -> str:
    return s.translate(_WX_TABLE)


def slp1_to_wx(s: str) -> str:
    return s.translate(_SLP1_WX_TABLE)


# --------------------------------------------------------------------------
# IAST <-> SLP1
# --------------------------------------------------------------------------
_IAST2SLP1 = {
    "a": "a", "ā": "A", "i": "i", "ī": "I", "u": "u", "ū": "U",
    "ṛ": "f", "ṝ": "F", "ḷ": "x", "ḹ": "X",
    "e": "e", "ai": "E", "o": "o", "au": "O",
    "ṃ": "M", "ṁ": "M", "ḥ": "H", "m̐": "~", "'": "'", "’": "'",
    "k": "k", "kh": "K", "g": "g", "gh": "G", "ṅ": "N",
    "c": "c", "ch": "C", "j": "j", "jh": "J", "ñ": "Y",
    "ṭ": "w", "ṭh": "W", "ḍ": "q", "ḍh": "Q", "ṇ": "R",
    "t": "t", "th": "T", "d": "d", "dh": "D", "n": "n",
    "p": "p", "ph": "P", "b": "b", "bh": "B", "m": "m",
    "y": "y", "r": "r", "l": "l", "v": "v",
    "ś": "S", "ṣ": "z", "s": "s", "h": "h",
    "ï": "i", "ü": "u",   # diaeresis marks hiatus: a+ï, a+ü
}
_IAST_KEYS = sorted(_IAST2SLP1, key=len, reverse=True)
_SLP12IAST = {}
for _k, _v in _IAST2SLP1.items():
    # prefer the canonical spelling (first listed; skip variants)
    if _v not in _SLP12IAST and _k not in ("ṁ", "’", "ï", "ü"):
        _SLP12IAST[_v] = _k


def iast_to_slp1(s: str, strict: bool = False) -> str:
    """Greedy longest-match decoding of (NFC) IAST into SLP1.

    Characters outside the alphabet (digits, punctuation, '-') are copied
    unless strict=True, in which case a ValueError is raised.
    """
    s = unicodedata.normalize("NFC", s.lower())
    out, i, n = [], 0, len(s)
    while i < n:
        for k in _IAST_KEYS:
            if s.startswith(k, i):
                out.append(_IAST2SLP1[k])
                i += len(k)
                break
        else:
            if strict:
                raise ValueError(f"non-IAST character {s[i]!r} in {s!r}")
            out.append(s[i])
            i += 1
    return "".join(out)


def slp1_to_iast(s: str) -> str:
    out = []
    for i, ch in enumerate(s):
        t = _SLP12IAST.get(ch, ch)
        # re-insert diaeresis where a + i/u would otherwise read as ai/au
        if ch in "iu" and i > 0 and s[i - 1] == "a":
            t = "ï" if ch == "i" else "ü"
        out.append(t)
    return "".join(out)


# --------------------------------------------------------------------------
# Devanagari -> SLP1 (abugida: consonants carry an inherent 'a')
# --------------------------------------------------------------------------
_DEV_VOWELS = dict(zip("अआइईउऊऋॠऌॡएऐओऔ", "aAiIuUfFxXeEoO"))
_DEV_MATRAS = dict(zip("ािीुूृॄॢॣेैोौ", "AiIuUfFxXeEoO"))
_DEV_CONS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
                     "kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshL"))
_DEV_OTHER = {"ं": "M", "ः": "H", "ँ": "~", "ऽ": "'"}
_VIRAMA = "्"
_NUKTA = "़"


def deva_to_slp1(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace(_NUKTA, "")
    out, pending_a = [], False
    for ch in s:
        if ch in _DEV_CONS:
            if pending_a:
                out.append("a")
            out.append(_DEV_CONS[ch])
            pending_a = True
        elif ch in _DEV_MATRAS:
            out.append(_DEV_MATRAS[ch])
            pending_a = False
        elif ch == _VIRAMA:
            pending_a = False
        else:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_DEV_VOWELS.get(ch) or _DEV_OTHER.get(ch, ch))
    if pending_a:
        out.append("a")
    return "".join(out)


# --------------------------------------------------------------------------
# Scheme detection (for auditing benchmark files)
# --------------------------------------------------------------------------
def detect_scheme(words) -> dict[str, float]:
    """Return log-likelihood-ratio style evidence for WX vs SLP1.

    Discriminating letters: 'z' (SLP1 retroflex sibilant, absent in WX except
    as rare candrabindu) and 'x' (WX dental d, very frequent; SLP1 vocalic l,
    near-absent). Returns the fraction of word types containing each.
    """
    words = [w for w in words if w]
    n = max(len(words), 1)
    fz = sum("z" in w for w in words) / n
    fx = sum("x" in w for w in words) / n
    guess = "WX" if fx > 5 * fz else "SLP1" if fz > 5 * fx else "unknown"
    return {"frac_z": fz, "frac_x": fx, "guess": guess}
