r"""Linguistically motivated normalisation profiles for the lemma views.

A *profile* is a string of flags, applied uniformly to every token of the
corpus (and, for orthography, to every benchmark word), never to benchmark
words alone:

  n  nasal canonicalisation   anusvara before a stop -> homorganic nasal
                              (saMDi -> sanDi, SaMkara -> SaNkara). Purely
                              orthographic: the nasal's place of articulation
                              is predictable from the following stop, so no
                              information is lost.
  p  participles as stems     kft participles and gerundives (VerbForm=Part/Gdv)
                              are represented by their nominal stem
                              (pratipadika) instead of the verbal root DCS uses:
                              gataH -> gata, pibantaH -> pibat, kurvARaH ->
                              kurvARa. The root's LemmaId is kept in the ID
                              views (gata_<id of gam>), so the link to the verb
                              is not lost.
  c  lexicalised compounds    adjacent compound members whose association
                              inside compounds is strong (NPMI >= theta,
                              count >= k) are fused into one token; productive,
                              compositional compounds stay split.

Linguistic basis. Panini defines the nominal stem (pratipadika) as the unit to
which case endings attach, and A. 1.2.46 (krt-taddhita-samasas ca) counts
krt-derivatives (participles included) and compounds as pratipadikas. "Remove
inflection, keep derivation" is therefore the consistent lemma: DCS's
root-lemmatisation of participles removes derivation as well. For compounds,
treating every samasa as a token would be faithful to 1.2.46 but destroys
distributional evidence for the productive majority, so only lexicalised ones
(detected by PMI, as in word2vec phrase detection) are fused.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_STOP_CLASS = {**dict.fromkeys("kKgG", "N"), **dict.fromkeys("cCjJ", "Y"),
               **dict.fromkeys("wWqQ", "R"), **dict.fromkeys("tTdD", "n"),
               **dict.fromkeys("pPbB", "m")}
_ANUSVARA = re.compile(r"M(?=[kKgGcCjJwWqQtTdDpPbB])")


def canon_nasal(s: str) -> str:
    """Anusvara before a stop -> the stop's homorganic nasal (SLP1)."""
    return _ANUSVARA.sub(lambda m: _STOP_CLASS[s[m.end()]], s)


# --------------------------------------------------------------- participles
# endings (SLP1) by stem class, longest first; base = form[:-len(ending)]
_A = ["AByAm", "eByaH", "ABiH", "AByaH", "AyAH", "AyAm", "asya", "ayoH", "AnAm", "ena",
      "Aya", "ezu", "Ani", "ayA", "AyE", "Asu", "aH", "am", "An", "EH", "At", "Am", "AH",
      "O", "e", "E", "A", "a"]
_AT = ["adByAm", "adByaH", "adBiH", "antaH", "antam", "antO", "anti", "atAm", "ataH",
       "atam", "atO", "atsu", "atA", "ate", "ati", "atI", "an", "at"]
# a-roots (yA, BA, ...): strong stem -Ant-, weak -At-  (yAntaH -> yAt)
_AAT = ["AntaH", "Antam", "AntO", "Anti", "AtAm", "AtaH", "AtA", "Ate", "Ati", "An", "At"]
_FEM_I = ["IByAm", "IByaH", "IBiH", "InAm", "yAH", "yAm", "yA", "yE", "yO", "yaH", "Izu", "Im", "IH", "I", "i"]
_ATI = [p + e for p in ("ant", "at") for e in _FEM_I]          # feminine -antI / -atI
_VAT = ["vadByAm", "vadByaH", "vadBiH", "vantaH", "vantam", "vantO", "vanti", "vatAm",
        "vataH", "vatsu", "vatA", "vate", "vati", "vAn", "van", "vat"] + ["vat" + e for e in _FEM_I]
for _l in (_A, _AT, _AAT, _ATI, _VAT):
    _l.sort(key=len, reverse=True)

_MIDDLE = ("mAna", "mARa", "Ana", "ARa")


def _strip(form, endings, add):
    for e in endings:
        if form.endswith(e) and len(form) > len(e):          # base of >= 1 letter (sat, sati)
            return form[: -len(e)] + add
    return None


def participle_stem(form: str, feats: str) -> str | None:
    """Nominal stem of a participle/gerundive from its unsandhied SLP1 form.

    Compound members (Case=Cpd) already carry the stem. Otherwise the stem
    class expected from the morphological features is tried first, then the
    remaining classes, so every participle is treated by the same rule set:
      past (no Tense, or Tense=Past)  kta a-stem; ktavatu vat-stem if -ta/-na + vat
      present / future                middle -mAna/-Ana a-stem; active at-stem
      gerundive                       a-stem
    Feminine forms map to the masculine/neuter stem, as in dictionary
    headwords. Returns None only if no ending of any class matches.
    """
    if "Case=Cpd" in feats:
        return form
    a_st = _strip(form, _A, "a")
    at_st = (_strip(form, _ATI, "at") or _strip(form, _AAT, "At")
             if re.search(r"A(nt|t)", form[-6:]) else None) or _strip(form, _ATI, "at") \
        or _strip(form, _AT, "at")
    vat_st = _strip(form, _VAT, "vat") if re.search(r"[tn]av(An|an|ant|at|ad)", form) else None
    if "VerbForm=Gdv" in feats:
        order = (a_st, at_st)
    elif "Tense=" not in feats or "Tense=Past" in feats:
        order = (vat_st, a_st, at_st)
    else:
        middle = a_st if a_st and a_st.endswith(_MIDDLE) else None
        aat = _strip(form, _AAT, "At")
        order = (middle, aat if aat and not middle else None, at_st, a_st)
    return next((x for x in order if x), None)


# ----------------------------------------------------------------- compounds
def join(x: str, y: str) -> str:
    """Forward vowel / common consonant sandhi at a compound junction (SLP1)."""
    if not x or not y:
        return x + y
    l, r = x[-1], y[0]
    if l in "aA" and r in "aA": return x[:-1] + "A" + y[1:]
    if l in "aA" and r in "iI": return x[:-1] + "e" + y[1:]
    if l in "aA" and r in "uU": return x[:-1] + "o" + y[1:]
    if l in "aA" and r in "eE": return x[:-1] + "E" + y[1:]
    if l in "aA" and r in "oO": return x[:-1] + "O" + y[1:]
    if l in "aA" and r == "f": return x[:-1] + "ar" + y[1:]
    if l in "iI" and r in "iI": return x[:-1] + "I" + y[1:]
    if l in "uU" and r in "uU": return x[:-1] + "U" + y[1:]
    if l in "iI" and r in "aAuUeEoOf": return x[:-1] + "y" + y
    if l in "uU" and r in "aAiIeEoOf": return x[:-1] + "v" + y
    if l in "tkwp" and r in "gGdDbBjJqQyrlvaAiIuUeEoOf":
        return x[:-1] + {"t": "d", "k": "g", "w": "q", "p": "b"}[l] + y
    if x.endswith("as") and r in "gGdDbBjJqQyrlvmn": return x[:-2] + "o" + y
    return x + y


class CompoundLexicon:
    """Decide which adjacent compound-member pairs are lexicalised.

    Statistics are collected over adjacent pairs (a, b) *inside compounds*
    (a = member lemma token, b = next member's or the head's lemma token):
        NPMI(a, b) = log2[ P(a, b) / (P(a .) P(. b)) ] / -log2 P(a, b)  in [-1, 1].
    A pair is fused iff count(a, b) >= min_count and NPMI >= theta. Both
    thresholds are fixed a priori (not tuned on any benchmark).
    """
    def __init__(self, min_count: int = 10, theta: float = 0.5):
        self.pairs, self.left, self.right = Counter(), Counter(), Counter()
        self.min_count, self.theta = min_count, theta
        self.N = 0

    def add(self, a: str, b: str):
        self.pairs[(a, b)] += 1
        self.left[a] += 1
        self.right[b] += 1
        self.N += 1

    def npmi(self, a: str, b: str) -> float:
        c = self.pairs.get((a, b), 0)
        if c == 0:
            return -1.0
        pab = c / self.N
        pmi = math.log2(pab / ((self.left[a] / self.N) * (self.right[b] / self.N)))
        return pmi / -math.log2(pab)

    def fuse(self, a: str, b: str) -> bool:
        return self.pairs.get((a, b), 0) >= self.min_count and self.npmi(a, b) >= self.theta

