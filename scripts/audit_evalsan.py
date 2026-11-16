"""Word-level transliteration audit and leak checks for EvalSan (non-destructive).

usage: python scripts/audit_evalsan.py --evalsan DIR --amara CSV [--out data/evalsan_audit]

For every benchmark word w (as written in the file) we consider two readings,
    r_S = w            (w read as SLP1)
    r_W = wx_to_slp1(w) (w read as WX, converted to SLP1)
and decide between them with two kinds of evidence:
  1. dictionary lookup in a lexicon L = Amarakosa words (the source EvalSan was
     built from) U DCS lemma strings, both converted to SLP1 independently;
  2. a character trigram language model P trained on DCS lemma strings: the
     log-likelihood ratio  LLR(w) = log2 P(r_W) - log2 P(r_S)  (bits).
     Positive = the WX reading is more Sanskrit-like.
Classes per word:
  invariant   r_S == r_W (no scheme-specific letter; reading irrelevant)
  wx          only r_W in L            slp1   only r_S in L
  both        both readings in L (true ambiguity; LLR breaks the tie)
  neither     neither in L (LLR decides, flagged low-confidence)
Writes per-word decisions (words.tsv), a per-file summary (summary.json), and
leak/duplicate checks (leaks.json). Nothing in EvalSan is modified.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import benchmarks as bm  # noqa: E402
from vidruma.translit import wx_to_slp1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--evalsan", type=Path, required=True)
ap.add_argument("--amara", type=Path, required=True)
ap.add_argument("--corpora", type=Path, default=ROOT / "data" / "corpora" / "all")
ap.add_argument("--out", type=Path, default=ROOT / "data" / "evalsan_audit")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- lexicon
ak = {e.word for e in bm.load_amarakosa(a.amara)}
lem = Counter()
for line in open(a.corpora / "lem.txt", encoding="utf-8"):
    lem.update(line.split())
seg = Counter()   # inflected forms, needed for the syntactic categorisation file
for line in open(a.corpora / "seg.txt", encoding="utf-8"):
    seg.update(line.split())
L = ak | set(lem) | {w for w, c in seg.items() if c >= 2}


# --------------------------------------------- character trigram LM (bits)
class CharLM:
    """Add-k smoothed character trigram model over SLP1 strings with ^ $ padding."""
    def __init__(self, words, k=0.1):
        self.k, self.tri, self.bi = k, Counter(), Counter()
        self.V = len({c for w in words for c in w}) + 1
        for w in words:
            s = "^^" + w + "$"
            for i in range(2, len(s)):
                self.tri[s[i - 2:i + 1]] += 1
                self.bi[s[i - 2:i]] += 1

    def logp(self, w):
        s = "^^" + w + "$"
        return sum(math.log2((self.tri[s[i - 2:i + 1]] + self.k) / (self.bi[s[i - 2:i]] + self.k * self.V))
                   for i in range(2, len(s)))


lm = CharLM(list(lem))   # types, not tokens: spelling, not frequency


def decide(w):
    rS, rW = w, wx_to_slp1(w)
    llr = lm.logp(rW) - lm.logp(rS)
    if rS == rW:
        return "invariant", rS, 0.0
    inS, inW = rS in L, rW in L
    if inW and not inS:
        return "wx", rW, llr
    if inS and not inW:
        return "slp1", rS, llr
    cls = "both" if inS else "neither"
    return cls, (rW if llr > 0 else rS), llr


# ------------------------------------------------------------ the files
FILES = {
    "rel_dev": ("automated_relatedness_AK_dev.csv", lambda r: r[:2]),
    "rel_test": ("automated_relatedness_AK_test.csv", lambda r: r[:2]),
    "syn_mcq": ("final_synonym_MCQs_AK.csv", lambda r: r[:5]),
    "cat_sem": ("final_semantic_categorization.csv", lambda r: [x for x in r[1:] if x]),
    "cat_syn": ("final_syntactic_categorization.csv", lambda r: [x for x in r if x]),
}
rows_by = {}
for key, (fn, get) in FILES.items():
    rows = [r for r in csv.reader(open(a.evalsan / fn, encoding="utf-8")) if r]
    if key == "syn_mcq":
        rows = rows[1:]
    rows_by[key] = (rows, get)

summary, words_out = {}, []
for key, (rows, get) in rows_by.items():
    cls = Counter()
    row_mix = Counter()
    for r in rows:
        kinds = set()
        for w in get(r):
            c, res, llr = decide(w)
            cls[c] += 1
            words_out.append((key, w, c, res, f"{llr:.2f}"))
            if c in ("wx", "slp1"):
                kinds.add(c)
            elif c in ("both", "neither"):
                kinds.add("wx?" if llr > 0 else "slp1?")
        row_mix["mixed_certain" if {"wx", "slp1"} <= kinds else
                "mixed_uncertain" if len({k.rstrip("?") for k in kinds}) > 1 else "consistent"] += 1
    n = sum(cls.values())
    summary[key] = {"tokens": n, **{k: v for k, v in cls.items()},
                    "rows": len(rows), **{f"rows_{k}": v for k, v in row_mix.items()}}

with open(a.out / "words.tsv", "w", encoding="utf-8") as fh:
    fh.write("file\tword\tclass\tresolved_slp1\tllr_wx_minus_slp1_bits\n")
    for t in words_out:
        fh.write("\t".join(t) + "\n")
json.dump(summary, open(a.out / "summary.json", "w"), indent=1)

# ------------------------------------------------------------ leak checks
res = {w: d[3] for d in words_out for w in [d[1]]}
def pairs(key):
    return [(res[r[0]], res[r[1]], r[2]) for r in rows_by[key][0]]
dev, test = pairs("rel_dev"), pairs("rel_test")
und = lambda p: frozenset(p[:2])
dev_set, test_set = {und(p) for p in dev}, {und(p) for p in test}
lab = defaultdict(set)
for p in dev + test:
    lab[und(p)].add(p[2])
test_words = {w for p in test for w in p[:2]}
dev_words = {w for p in dev for w in p[:2]}
mcq = [[res[w] for w in r[:5]] for r in rows_by["syn_mcq"][0]]
mcq_words = {w for r in mcq for w in r}
leaks = {
    "rel_test_pairs": len(test), "rel_dev_pairs": len(dev),
    "dev_test_pair_overlap": len(dev_set & test_set),
    "dev_test_word_overlap_frac_of_test_words": len(dev_words & test_words) / len(test_words),
    "duplicate_pairs_in_test": len(test) - len({und(p) for p in test}),
    "pairs_with_conflicting_labels": sum(len(v) > 1 for v in lab.values()),
    "self_pairs": sum(p[0] == p[1] for p in dev + test),
    "mcq_items": len(mcq), "mcq_duplicate_items": len(mcq) - len({tuple(r) for r in mcq}),
    "mcq_items_with_repeated_word": sum(len(set(r)) < 5 for r in mcq),
    "mcq_words_also_in_rel_test_frac": len(mcq_words & test_words) / len(mcq_words),
}
# MCQ gold verification against the Amarakosa synsets
w2v, w2s = bm.ak_partitions(bm.load_amarakosa(a.amara))
gold_ok = sum(bool(w2s.get(r[0], set()) & w2s.get(r[1], set())) for r in mcq)
amb = sum(any(w2s.get(r[0], set()) & w2s.get(d, set()) for d in r[2:]) for r in mcq)
leaks.update({"mcq_gold_shares_AK_synset": gold_ok, "mcq_items_with_synonymous_distractor": amb})
json.dump(leaks, open(a.out / "leaks.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(json.dumps(leaks, indent=1))
