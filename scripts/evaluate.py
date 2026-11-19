"""Evaluate a trained run under one or more protocols / plugins.

usage: python scripts/evaluate.py results/<subset>/<run> --evalsan DIR --amara CSV \
           [--protocols ours evalsan evalsan_raw ours+subword] [--plugins compose]

Each protocol / plugin writes its OWN file in the run directory:
    eval_<protocol>.json         summary metrics
    items_<protocol>.json        per-item scores (for common-support tables
                                 and paired tests in make_tables.py)
so any protocol or plugin can be dropped by not selecting / deleting its files.

Protocols
  ours          OOV excluded + coverage; rank statistics; dev-tuned thresholds;
                permutation nulls (vidruma.evaluate)
  ours+center   as `ours`, after mean-centring the space (removes the common
                direction behind anisotropy; Mu & Viswanath 2018 without PC removal)
  ours+subword  as `ours`, plus FastText subword vectors for unseen strings
  evalsan       EvalSan-faithful (vidruma.evalsan_compat), scheme-normalised
  evalsan_raw   EvalSan-faithful with the original, unconverted WX words
Plugins
  compose       `ours` on a space augmented with composed vectors for words
                approved in data/compose/<subset>/candidates.tsv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import benchmarks as bm  # noqa: E402
from vidruma import evaluate as ev  # noqa: E402
from vidruma.embed import Space  # noqa: E402

EXCL = {"nAnArTavargaH", "avyayavargaH"}   # polysemous / indeclinable vargas

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--evalsan", type=Path, required=True)
ap.add_argument("--amara", type=Path, required=True)
ap.add_argument("--protocols", nargs="+", default=["ours"],
                choices=["ours", "ours+center", "ours+subword", "evalsan", "evalsan_raw"])
ap.add_argument("--plugins", nargs="*", default=[], choices=["compose"])
ap.add_argument("--compose_tsv", type=Path, help="default data/compose/<subset>/candidates.tsv")
ap.add_argument("--compose_accept", default="approved",
                help="comma-separated statuses to accept, e.g. approved,auto")
ap.add_argument("--lookup", default="maxsim", choices=["maxsim", "maxfreq", "mean"])
ap.add_argument("--n_perm", type=int, default=200)
ap.add_argument("--bench_tag", default="", help="suffix for result files when --evalsan is not the "
                "original EvalSan, e.g. 'clean' -> eval_ours@clean.json")
ap.add_argument("--respell", action="store_true",
                help="map OOV benchmark words to in-vocabulary spelling variants "
                     "(normalise.Respeller); result files get a '+spell' suffix")
a = ap.parse_args()

space = Space.load(a.run / "vectors.npz")
subset = a.run.parent.name
profile = subset.split("+", 1)[1] if "+" in subset else ""
if "n" in profile:          # corpus orthography -> apply the same rule to benchmark words
    from vidruma.normalise import canon_nasal
    space.canon = canon_nasal
if a.respell:                # OOV benchmark words -> in-vocabulary spelling variant
    space.use_respelling()
TAG = ("+spell" if a.respell else "") + (f"@{a.bench_tag}" if a.bench_tag else "")
how_vec = "maxfreq" if a.lookup == "maxsim" else a.lookup


def ours(sp, tag):
    R, items = {"vocab": len(sp.words)}, {}
    dev, test = bm.load_relatedness(a.evalsan, "dev"), bm.load_relatedness(a.evalsan, "test")
    _, (sd, yd) = ev.relatedness(sp, dev, a.lookup)
    R["relatedness_test"], (st, yt) = ev.relatedness(sp, test, a.lookup)
    if len(sd) > 10:
        _, th = ev.tune_thresholds(sd, yd)
        R["relatedness_test"]["macroF1_3way"] = ev.classify_relatedness(st, yt, th)
        R["relatedness_test"]["thresholds_dev"] = [float(t) for t in th]
    items["relatedness_test"] = {i: sp.sim(x, y, a.lookup) for i, (x, y, _) in enumerate(test)
                                 if x in sp and y in sp}
    mcq = bm.load_synonym_mcq(a.evalsan)
    R["synonym_mcq"] = ev.synonym_mcq(sp, mcq, a.lookup)
    items["synonym_mcq"] = {i: int(np.argmax([sp.sim(q, x, a.lookup) for x in o]) == 0)
                            for i, (q, o) in enumerate(mcq)
                            if q in sp and all(x in sp for x in o)}
    for kind in ("sem", "syn"):
        cats = bm.load_categorisation(a.evalsan, kind)
        pairs = [(w, c) for c, ws in cats.items() for w in ws if w in sp]
        if len(pairs) > len(cats):
            X = np.stack([sp.vec(w, how_vec) for w, _ in pairs])
            r, _ = ev.cluster_scores(X, [c for _, c in pairs], k=len(cats), n_perm=a.n_perm)
            r["coverage"] = len(pairs) / sum(map(len, cats.values()))
            R[f"cat_{kind}"] = r
    # RQ2: Amarakosa vargas
    w2v, _ = bm.ak_partitions(bm.load_amarakosa(a.amara))
    elig = [w for w, vs in w2v.items() if len(vs) == 1 and not vs & EXCL]
    uniq = sorted(w for w in elig if w in sp)
    gold = [next(iter(w2v[w])) for w in uniq]
    X = np.stack([sp.vec(w, how_vec) for w in uniq])
    for m in ("kmeans", "ward"):
        R[f"ak_varga_{m}"], lab = ev.cluster_scores(X, gold, method=m, n_perm=a.n_perm)
        R[f"ak_varga_{m}"]["coverage"] = len(uniq) / len(elig)
        items[f"ak_varga_{m}"] = dict(zip(uniq, map(int, lab)))
    return R, items


def dump(tag, R, items=None):
    R = {"protocol": tag, "subset": subset, "run": a.run.name, **R}
    json.dump(R, open(a.run / f"eval_{tag}{TAG}.json", "w"), indent=1, default=float)
    if items is not None:
        json.dump(items, open(a.run / f"items_{tag}{TAG}.json", "w"), default=float)
    print(f"[{tag}]", json.dumps(R, default=float)[:600], "...")


for p in a.protocols:
    if p == "ours":
        dump(p, *ours(space, p))
    elif p == "ours+center":
        cs = Space(space.words, space.X - space.X.mean(0), space.counts)
        cs.canon = space.canon
        dump(p, *ours(cs, p))
    elif p == "ours+subword":
        f = a.run / "subword_oov.npz"
        if not f.exists():
            print("ours+subword: no subword_oov.npz (only FastText runs have one); skipped")
            continue
        z = np.load(f)
        aug = Space(space.words + z["words"].tolist(), np.vstack([space.X, z["X"]]),
                    np.concatenate([space.counts, np.ones(len(z["words"]))]))
        aug.canon = space.canon
        dump(p, *ours(aug, p))
    else:
        from vidruma import evalsan_compat as ec
        scheme = "raw" if p == "evalsan_raw" else "auto"
        R = {"relatedness_test": ec.relatedness(space, bm.load_relatedness(a.evalsan, "test", scheme), how_vec),
             "synonym_mcq": ec.synonym_mcq(space, bm.load_synonym_mcq(a.evalsan, scheme), how_vec),
             "cat_sem": ec.categorisation(space, bm.load_categorisation(a.evalsan, "sem", scheme), how_vec),
             "cat_syn": ec.categorisation(space, bm.load_categorisation(a.evalsan, "syn", scheme), how_vec)}
        dump(p, R)

if "compose" in a.plugins:
    from vidruma.plugins import compose
    tsv = a.compose_tsv or ROOT / "data" / "compose" / subset / "candidates.tsv"
    acc = compose.read_accepted(tsv, tuple(a.compose_accept.split(",")))
    aug, added = compose.augment(space, acc)
    aug.canon = space.canon
    R, items = ours(aug, "plugin_compose")
    R["compose"] = {"tsv": str(tsv), "accepted_statuses": a.compose_accept,
                    "accepted_rows": len(acc), "vectors_added": len(added)}
    dump("plugin_compose", R, items)
