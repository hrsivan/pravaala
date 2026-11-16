"""Build a cleaned copy of EvalSan's intrinsic benchmarks (EvalSan is not modified).

usage: python scripts/clean_evalsan.py --evalsan DIR --amara CSV [--out data/evalsan_clean]

Output files keep EvalSan's names and formats, so every loader works unchanged
(point --evalsan at the output directory). All words are written in SLP1.

Rules (each removal is counted in CLEANING.md):
  encoding   each file converted as a whole by its detected scheme (WX -> SLP1
             for the Amarakosa-derived files). A word-level audit
             (scripts/audit_evalsan.py) found no file mixing schemes; words
             whose two readings are both real words follow their file.
  relatedness  drop self-pairs (u, u); drop exact duplicates (unordered) within
             a split; drop pairs that carry different labels anywhere; drop from
             DEV any pair that also occurs in TEST (test is never shrunk for this).
  synonym MCQ  drop items where the question is its own answer, where the answer
             is repeated among the distractors, where any word repeats, or where
             a distractor shares an Amarakosa meaning (artha) with the question
             (two correct answers).
  categorisation  copied unchanged (already SLP1).
"""
import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import benchmarks as bm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--evalsan", type=Path, required=True)
ap.add_argument("--amara", type=Path, required=True)
ap.add_argument("--out", type=Path, default=ROOT / "data" / "evalsan_clean")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
log = Counter()

# ------------------------------------------------------------ relatedness
dev, test = bm.load_relatedness(a.evalsan, "dev"), bm.load_relatedness(a.evalsan, "test")
key = lambda u, v: frozenset((u, v))
labels = defaultdict(set)
for u, v, y in dev + test:
    labels[key(u, v)].add(y)
conflict = {k for k, ys in labels.items() if len(ys) > 1}
test_keys = {key(u, v) for u, v, _ in test}


def clean_pairs(pairs, split):
    seen, out = set(), []
    for u, v, y in pairs:
        k = key(u, v)
        if u == v:
            log[f"{split}: self-pair"] += 1; continue
        if k in conflict:
            log[f"{split}: conflicting label"] += 1; continue
        if k in seen:
            log[f"{split}: duplicate"] += 1; continue
        if split == "dev" and k in test_keys:
            log["dev: also in test"] += 1; continue
        seen.add(k)
        out.append((u, v, y))
    return out


for split, pairs in (("dev", dev), ("test", test)):
    kept = clean_pairs(pairs, split)
    log[f"{split}: kept"] = len(kept)
    with open(a.out / bm.FILES[f"rel_{split}"], "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(kept)

# ------------------------------------------------------------------- MCQ
w2a = defaultdict(set)
for e in bm.load_amarakosa(a.amara):
    w2a[e.word].add(e.artha)
header = next(csv.reader(open(a.evalsan / bm.FILES["syn_mcq"], encoding="utf-8")))
kept = []
for q, opts in bm.load_synonym_mcq(a.evalsan):
    ans, dis = opts[0], opts[1:]
    if q == ans:
        log["mcq: question is its own answer"] += 1; continue
    if ans in dis:
        log["mcq: answer repeated among distractors"] += 1; continue
    if len({q, *opts}) < 5:
        log["mcq: other repeated word"] += 1; continue
    if any(w2a[q] & w2a[d] for d in dis):
        log["mcq: a distractor is also a synonym"] += 1; continue
    kept.append([q, *opts])
log["mcq: kept"] = len(kept)
with open(a.out / bm.FILES["syn_mcq"], "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    w.writerows(kept)

# ------------------------------------------------------- categorisation
for k in ("cat_sem", "cat_syn"):
    shutil.copy(a.evalsan / bm.FILES[k], a.out / bm.FILES[k])

with open(a.out / "CLEANING.md", "w", encoding="utf-8") as fh:
    fh.write("# EvalSan, cleaned\n\nBuilt by scripts/clean_evalsan.py from EvalSan's "
             "evaluations/Intrinsic/Data. All words SLP1.\n\n| step | count |\n|---|---|\n")
    for k in sorted(log):
        fh.write(f"| {k} | {log[k]} |\n")
for k in sorted(log):
    print(f"{k:45s} {log[k]}")
