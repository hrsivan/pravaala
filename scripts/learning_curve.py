"""Does more data help? PPMI-SVD trained on nested fractions of the corpus.

usage: python scripts/learning_curve.py [--subset all_nolex+np] [--evalsan data/evalsan_clean]
Sentences are shuffled once (seed 0); fraction f uses the first f of them, so
the samples are nested. All fractions are scored on the items covered by the
SMALLEST sample (common support), so the curve measures vector quality, not
coverage; coverage is reported separately. Writes results/<subset>/learning_curve@<tag>.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import benchmarks as bm, embed, evaluate as ev  # noqa: E402
from vidruma.normalise import canon_nasal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all_nolex+np")
ap.add_argument("--evalsan", type=Path, default=ROOT / "data" / "evalsan_clean")
ap.add_argument("--tag", default="clean")
ap.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
ap.add_argument("--dim", type=int, default=300)
a = ap.parse_args()

lines = open(ROOT / "data" / "corpora" / a.subset / "lem.txt", encoding="utf-8").read().split("\n")
random.Random(0).shuffle(lines)
canon = canon_nasal if "n" in (a.subset.split("+", 1) + [""])[1] else None
test = bm.load_relatedness(a.evalsan, "test")
mcq = bm.load_synonym_mcq(a.evalsan)
spaces = {}
for frac in a.fractions:
    part = lines[: int(len(lines) * frac)]
    sents = lambda: (l.split() for l in part)
    vocab, counts = embed.build_vocab(sents, 5)
    X = embed.svd_embeddings(embed.cooccurrence(sents, vocab, 5), a.dim, p=0.5)
    s = embed.Space(list(vocab), X, counts)
    s.canon = canon
    spaces[frac] = (s, sum(len(l.split()) for l in part))
    print(f"trained {frac}: {spaces[frac][1]} tokens, vocab {len(vocab)}", flush=True)

small = spaces[min(a.fractions)][0]
rel = [(u, v, y) for u, v, y in test if u in small and v in small]
items = [(q, o) for q, o in mcq if q in small and all(x in small for x in o)]
out = {"common_rel_n": len(rel), "common_mcq_n": len(items), "points": []}
for frac, (s, ntok) in spaces.items():
    r, (sc, y) = ev.relatedness(s, rel)
    m = ev.synonym_mcq(s, items)
    full_r, _ = ev.relatedness(s, test)
    full_m = ev.synonym_mcq(s, mcq)
    pt = {"fraction": frac, "tokens": ntok, "rho_common": r["spearman"], "auc20_common": r["auc_2v0"],
          "mcq_common": m["acc"], "rel_coverage": full_r["coverage"], "mcq_coverage": full_m["coverage"]}
    out["points"].append(pt)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in pt.items()}, flush=True)
json.dump(out, open(ROOT / "results" / a.subset / f"learning_curve@{a.tag}.json", "w"), indent=1)
