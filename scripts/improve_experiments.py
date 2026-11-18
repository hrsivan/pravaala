"""Can better technique raise the scores? Dev-selected experiments, test reported once.

usage: python scripts/improve_experiments.py [--subset all_nolex+np] [--evalsan data/evalsan_clean]
Selection data : relatedness DEV split + even-indexed synonym items.
Report data    : relatedness TEST split + odd-indexed synonym items (held out).
Corpus/view    : <subset>/lem (default: leak-free, nasal + participle stems).

Experiments (each compared with the same-family baseline):
  A  PPMI-SVD hyperparameters: shift k in {1, 5}, eigenweight p, w+c
     (add context vectors V Sigma^p to word vectors U Sigma^p; Levy et al. 2015)
  B  post-processing: centre + remove the top D principal components
     (all-but-the-top, Mu & Viswanath 2018), D chosen on dev
  C  SGNS trained longer: 15 epochs, 10 negatives (vs 5 / 5)
  D  dependency contexts: add (dependent, head) pairs from DCS's parsed
     portion (~29% of tokens) to the linear co-occurrence counts, weight lambda
Writes results/<subset>/improve@<tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import paths  # noqa: E402
from vidruma import benchmarks as bm, embed, evaluate as ev  # noqa: E402
from vidruma.dcs import iter_corpus, load_time_slots, view  # noqa: E402
from vidruma.infotheory import ppmi_matrix  # noqa: E402
from vidruma.normalise import canon_nasal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all_nolex+np")
ap.add_argument("--evalsan", type=Path, default=ROOT / "data" / "evalsan_clean")
ap.add_argument("--tag", default="clean")
ap.add_argument("--conllu", type=Path, default=paths.DCS_CONLLU)
ap.add_argument("--dim", type=int, default=300)
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

base_subset, profile = (a.subset.split("+", 1) + [""])[:2]
CANON = canon_nasal if "n" in profile else None
corpus = ROOT / "data" / "corpora" / a.subset / "lem.txt"
sents = lambda: (l.split() for l in open(corpus, encoding="utf-8"))
vocab, counts = embed.build_vocab(sents, 5)
words = list(vocab)
dev, test = bm.load_relatedness(a.evalsan, "dev"), bm.load_relatedness(a.evalsan, "test")
mcq = bm.load_synonym_mcq(a.evalsan)
mcq_dev, mcq_test = mcq[0::2], mcq[1::2]
R = {}


def space(X, w=words, c=counts):
    s = embed.Space(w, X, c)
    s.canon = CANON
    return s


def dev_score(s):
    r, _ = ev.relatedness(s, dev)
    m = ev.synonym_mcq(s, mcq_dev)
    return r["spearman"] + m["acc"], r["spearman"], m["acc"]


def test_report(s, name):
    rt, (st, yt) = ev.relatedness(s, test)
    _, (sd, yd) = ev.relatedness(s, dev)
    _, th = ev.tune_thresholds(sd, yd)
    m = ev.synonym_mcq(s, mcq_test)
    R[name] = {"rho": rt["spearman"], "auc20": rt["auc_2v0"], "auc10": rt["auc_1v0"],
               "f1_dev_thr": ev.classify_relatedness(st, yt, th), "rel_cov": rt["coverage"],
               "mcq_odd": m["acc"], "mcq_n": m["n"], "mcq_wilson": m["wilson95"]}
    _, dr, dm = dev_score(s)                      # the selection data, kept for the decision log
    R[name].update(dev_rho=dr, dev_mcq_even=dm)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[name].items()}, flush=True)


def center_abtt(X, D):
    X = X - X.mean(0)
    if D:
        _, _, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)],
                                 full_matrices=False)
        X = X - (X @ Vt[:D].T) @ Vt[:D]
    return X


C_lin = embed.cooccurrence(sents, vocab, window=a.window)


def svd_family(C, label):
    """Select shift k, eigenweight p, w+c and ABTT depth D on dev; report test."""
    best = None
    for k in (1, 5):
        M = ppmi_matrix(C, alpha=0.75, shift=k)
        U, S, Vt = svds(M, k=a.dim, v0=np.random.default_rng(0).standard_normal(min(M.shape)))
        o = np.argsort(-S)
        U, S, V = U[:, o], S[o], Vt[o].T
        for p in (0.0, 0.5):
            for wc in (False, True):
                X = U * S ** p + (V * S ** p if wc else 0)
                for D in (None, 0, 1, 3):          # None = raw; 0 = centred; D>0 = ABTT
                    sc = dev_score(space(X if D is None else center_abtt(X, D)))
                    cfg = {"shift": k, "p": p, "w+c": wc, "abtt_D": D}
                    print(f"  dev {label} {cfg} rho={sc[1]:.3f} mcq={sc[2]:.3f}", flush=True)
                    if best is None or sc[0] > best[0]:
                        best = (sc[0], cfg, X)
    cfg, X = best[1], best[2]
    R.setdefault("_selected", {})[label] = cfg
    D = cfg["abtt_D"]
    test_report(space(X if D is None else center_abtt(X, D)), label)


# ---- baseline (the configuration of the main grid's tuned model) -------------
X0 = embed.svd_embeddings(C_lin, a.dim, p=0.5)
test_report(space(X0), "PPMI-SVD baseline (k=1, p=0.5)")
# ---- A + B -----------------------------------------------------------------
svd_family(C_lin, "PPMI-SVD tuned (k, p, w+c, ABTT on dev)")

# ---- D: dependency contexts --------------------------------------------------
import json as _json  # noqa: E402
spec = _json.load(open(ROOT / "config" / "subsets.json"))[base_subset]
keep = set(spec["slots"]) if isinstance(spec, dict) else set(spec)
excl = set(spec.get("exclude_texts", [])) if isinstance(spec, dict) else set()
slots = load_time_slots(a.conllu / "lookup" / "chapter-info.xml")
rows, cols = [], []
n_arcs = 0
for s in iter_corpus(a.conllu / "files", slots, keep):
    if s.text in excl:
        continue
    toks = view(s, "lem", profile)
    if len(toks) != len(s.words):
        continue
    pos = {w.idx: i for i, w in enumerate(s.words)}
    for i, w in enumerate(s.words):
        if w.head > 0 and w.head in pos:
            d, h = toks[i], toks[pos[w.head]]
            if d in vocab and h in vocab:
                rows += [vocab[d], vocab[h]]
                cols += [vocab[h], vocab[d]]
                n_arcs += 1
C_dep = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=C_lin.shape).tocsr()
R["_dependency_arcs"] = n_arcs
print("dependency arcs:", n_arcs, flush=True)
best = None
for lam in (0.5, 1.0, 2.0):
    X = embed.svd_embeddings(C_lin + lam * C_dep, a.dim, p=0.5)
    sc = dev_score(space(X))
    print(f"  dev dependency lambda={lam} rho={sc[1]:.3f} mcq={sc[2]:.3f}", flush=True)
    if best is None or sc[0] > best[0]:
        best = (sc[0], lam, X)
R.setdefault("_selected", {})["dependency"] = {"lambda": best[1]}
test_report(space(best[2]), f"PPMI-SVD + dependency contexts (lambda={best[1]})")

# ---- C: SGNS trained properly ------------------------------------------------
for ep, neg in ((5, 5), (15, 10)):
    kv = embed.train_sgns(corpus, a.dim, a.window, 5, negative=neg, epochs=ep, workers=a.workers)
    s0 = embed.Space.from_kv(kv)
    s0.canon = CANON
    sc_raw, sc_c = dev_score(s0), dev_score(space(center_abtt(s0.X, 0), s0.words, s0.counts))
    use_c = sc_c[0] > sc_raw[0]
    s1 = space(center_abtt(s0.X, 0), s0.words, s0.counts) if use_c else s0
    test_report(s1, f"SGNS {ep} epochs, {neg} neg{' (centred, dev-chosen)' if use_c else ''}")

json.dump(R, open(ROOT / "results" / a.subset / f"improve@{a.tag}.json", "w"), indent=1, default=float)
print("WRITTEN")
