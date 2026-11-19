"""PMI smoothing for PPMI-SVD: absolute discounting and the Kneser-Ney
continuation marginal, crossed with the vocabulary threshold (min count).

Everything else is fixed at the dev-selected PPMI-SVD setting (window 10,
d=300, p=0.5). Each variant is scored
  * on its own coverage (lower min count -> more items covered), and
  * on the baseline support: the items the baseline (min count 5) covers,
    so that quality changes are not confounded with coverage changes.
Selection uses dev only (relatedness dev split, even MCQ half); test numbers
are computed for every variant but only the dev-selected ones are reported.

usage: python scripts/sweep_smoothing.py --subset all_nolex+np --tag clean \
           --evalsan data/evalsan_clean
Writes results/<subset>/sweep_smoothing_<view>@<tag>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import benchmarks as bm, embed, evaluate as ev  # noqa: E402
from vidruma.infotheory import ney_discount, ppmi_matrix  # noqa: E402
from vidruma.normalise import canon_nasal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all_nolex+np")
ap.add_argument("--view", default="lem")
ap.add_argument("--evalsan", type=Path, default=ROOT / "data" / "evalsan_clean")
ap.add_argument("--tag", default="clean")
ap.add_argument("--window", type=int, default=10)
ap.add_argument("--dim", type=int, default=300)
ap.add_argument("--p", type=float, default=0.5)
ap.add_argument("--min_counts", type=int, nargs="+", default=[5, 3, 2])
ap.add_argument("--variants", nargs="+", default=["a75", "a75+D", "cont", "cont+D"],
                help="a<alpha*100> or cont, optionally +D (Ney discount) or +D<value>")
a = ap.parse_args()

corpus = ROOT / "data" / "corpora" / a.subset / f"{a.view}.txt"
sents = lambda: (l.split() for l in open(corpus, encoding="utf-8"))
profile = a.subset.split("+", 1)[1] if "+" in a.subset else ""
CANON = canon_nasal if "n" in profile else None
cache = ROOT / "results" / a.subset / "cooc_cache"
cache.mkdir(parents=True, exist_ok=True)

dev, test = bm.load_relatedness(a.evalsan, "dev"), bm.load_relatedness(a.evalsan, "test")
mcq = bm.load_synonym_mcq(a.evalsan)
mcq_dev, mcq_test = mcq[0::2], mcq[1::2]


def parse(v):
    ctx, _, d = v.partition("+")
    alpha = 1.0
    if ctx.startswith("a"):
        alpha, ctx = int(ctx[1:]) / 100, "alpha"
    disc = 0.0 if not d else ("ney" if d == "D" else float(d[1:]))
    return ctx, alpha, disc


def scores(space, rel, items, rel_sub=None, items_sub=None):
    r, (s, y) = ev.relatedness(space, rel, "maxsim")
    m = ev.synonym_mcq(space, items, "maxsim")
    out = {"rel_cov": r["coverage"], "rho": r.get("spearman"), "auc20": r.get("auc_2v0"),
           "auc10": r.get("auc_1v0"), "mcq_cov": m["coverage"], "mcq": m["acc"], "mcq_n": m["n"]}
    if rel_sub is not None:
        rb, _ = ev.relatedness(space, rel_sub, "maxsim")
        mb = ev.synonym_mcq(space, items_sub, "maxsim")
        out.update({"base_rho": rb.get("spearman"), "base_auc20": rb.get("auc_2v0"),
                    "base_mcq": mb["acc"], "base_n": (rb["n"], mb["n"])})
    return out, (s, y)


rows, base_sets = [], None
outfile = ROOT / "results" / a.subset / f"sweep_smoothing_{a.view}@{a.tag}.json"
for mc in a.min_counts:
    vocab, counts = embed.build_vocab(sents, mc)
    f = cache / f"{a.view}_w{a.window}_m{mc}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        C = sp.csr_matrix((z["cd"], z["ci"], z["cp"]), shape=tuple(z["shape"]))
        R = sp.csr_matrix((z["rd"], z["ri"], z["rp"]), shape=tuple(z["shape"]))
    else:
        t0 = time.time()
        C, R = embed.cooccurrence(sents, vocab, window=a.window, unweighted=True)
        C.sort_indices(); R.sort_indices()
        np.savez(f, cd=C.data, ci=C.indices, cp=C.indptr, rd=R.data, ri=R.indices, rp=R.indptr,
                 shape=np.array(C.shape))
        print(f"cooc m{mc}: {time.time() - t0:.0f}s", flush=True)
    D_ney = ney_discount(R)
    stats = {"vocab": len(vocab), "nnz": int(C.nnz), "frac_r1": float(np.mean(np.isclose(R.data, 1))),
             "D_ney": D_ney}
    print(f"m{mc}", stats, flush=True)
    for v in a.variants:
        t0 = time.time()
        ctx, alpha, disc = parse(v)
        M = ppmi_matrix(C, alpha=alpha, R=R, discount=disc, ctx=ctx)
        U, S, _ = svds(M, k=a.dim, v0=np.random.default_rng(0).standard_normal(min(M.shape)))
        o = np.argsort(-S)
        space = embed.Space(list(vocab), U[:, o] * S[o] ** a.p, counts)
        space.canon = CANON
        if base_sets is None:   # first run = baseline (min count 5, first variant)
            base_sets = ([x for x in dev if x[0] in space and x[1] in space],
                         [x for x in test if x[0] in space and x[1] in space],
                         [x for x in mcq_dev if x[0] in space and all(o_ in space for o_ in x[1])],
                         [x for x in mcq_test if x[0] in space and all(o_ in space for o_ in x[1])])
        d, (sd, yd) = scores(space, dev, mcq_dev, base_sets[0], base_sets[2])
        t, (st, yt) = scores(space, test, mcq_test, base_sets[1], base_sets[3])
        f1dev, th = ev.tune_thresholds(sd, yd)
        d["f1"], t["f1"] = f1dev, ev.classify_relatedness(st, yt, th)
        row = {"min_count": mc, "variant": v, **stats, "pos_cells": int(M.nnz),
               "dev": d, "test": t, "secs": round(time.time() - t0)}
        rows.append(row)
        show = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in d.items()}
        print(f"m{mc} {v:8s} pos={M.nnz:>9d} DEV {show}", flush=True)
        json.dump({"config": vars(a) | {"evalsan": str(a.evalsan)}, "rows": rows},
                  open(outfile, "w"), indent=1, default=float)
print("done")
