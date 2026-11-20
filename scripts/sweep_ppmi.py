"""Dev-only hyperparameter sweep for PPMI-SVD (window, dim, eigenvalue weight p).

Selection uses only: relatedness DEV split, and the even-indexed half of the
synonym MCQ (EvalSan has no MCQ dev split). Test = relatedness test split and
odd-indexed MCQ half. usage: python scripts/sweep_ppmi.py [--view lem]
Writes results/<subset>/sweep_ppmi_<view>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala import paths  # noqa: E402
from pravaala import benchmarks as bm, embed, evaluate as ev  # noqa: E402
from pravaala.infotheory import ppmi_matrix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all")
ap.add_argument("--view", default="lem")
ap.add_argument("--evalsan", type=Path, default=paths.EVALSAN_ORIG)
ap.add_argument("--windows", type=int, nargs="+", default=[2, 5, 10, 20])
ap.add_argument("--dims", type=int, nargs="+", default=[100, 300])
ap.add_argument("--tag", default="", help="benchmark tag for the output name, e.g. clean")
ap.add_argument("--ps", type=float, nargs="+", default=[0.0, 0.5, 1.0])
a = ap.parse_args()

corpus = ROOT / "data" / "corpora" / a.subset / f"{a.view}.txt"
sents = lambda: (l.split() for l in open(corpus, encoding="utf-8"))
vocab, counts = embed.build_vocab(sents, 5)
from pravaala.normalise import canon_nasal  # noqa: E402
CANON = canon_nasal if "n" in (a.subset.split("+", 1)[1] if "+" in a.subset else "") else None
dev, test = bm.load_relatedness(a.evalsan, "dev"), bm.load_relatedness(a.evalsan, "test")
mcq = bm.load_synonym_mcq(a.evalsan)
mcq_dev, mcq_test = mcq[0::2], mcq[1::2]


def score(space, rel, items):
    space.canon = CANON
    r, (s, y) = ev.relatedness(space, rel, "maxsim")
    m = ev.synonym_mcq(space, items, "maxsim")
    return r, (s, y), m


out = []
for w in a.windows:
    C = embed.cooccurrence(sents, vocab, window=w)
    M = ppmi_matrix(C, alpha=0.75)
    U, S, _ = svds(M, k=max(a.dims), v0=np.random.default_rng(0).standard_normal(min(M.shape)))
    order = np.argsort(-S)
    U, S = U[:, order], S[order]
    for d in a.dims:
        for p in a.ps:
            sp = embed.Space(list(vocab), U[:, :d] * S[:d] ** p, counts)
            rd, (sd, yd), md = score(sp, dev, mcq_dev)
            f1dev, th = ev.tune_thresholds(sd, yd)
            row = {"window": w, "dim": d, "p": p, "dev_auc20": rd["auc_2v0"], "dev_auc10": rd["auc_1v0"],
                   "dev_rho": rd["spearman"], "dev_f1": f1dev, "dev_mcq": md["acc"], "th": th}
            out.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in row.items() if k != "th"}), flush=True)

# select on dev: mean of dev rho and dev MCQ accuracy
best = max(out, key=lambda r: r["dev_rho"] + r["dev_mcq"])
C = embed.cooccurrence(sents, vocab, window=best["window"])
X = embed.svd_embeddings(C, best["dim"], p=best["p"])
sp = embed.Space(list(vocab), X, counts)
rt, (st, yt), mt = score(sp, test, mcq_test)
res = {"selected": {k: best[k] for k in ("window", "dim", "p")},
       "test_rho": rt["spearman"], "test_auc20": rt["auc_2v0"], "test_auc10": rt["auc_1v0"],
       "test_f1_devthresholds": ev.classify_relatedness(st, yt, best["th"]),
       "test_mcq_odd_half": mt["acc"], "test_mcq_n": mt["n"], "sweep": out}
tag = f"@{a.tag}" if a.tag else ""
json.dump(res, open(ROOT / "results" / a.subset / f"sweep_ppmi_{a.view}{tag}.json", "w"), indent=1, default=float)
print("SELECTED", res["selected"], {k: round(v, 3) for k, v in res.items() if k.startswith("test_")})
