"""Decision log: the dev-set evidence behind each design decision.

Dev data = relatedness DEV split + even-indexed synonym MCQ items (EvalSan has
no MCQ dev split); the cleaned benchmark throughout. Within each decision the
options are compared on their *common support* (items every option covers),
so quality differences are not confounded with coverage; coverage itself is
reported separately (relatedness dev pairs, MCQ dev items).

Some decisions come from earlier dev-only sweeps whose JSONs are read here:
sweep_ppmi (window/dim/p), improve (technique), sweep_smoothing (smoothing and
min count). Nothing here looks at test data.

usage: python scripts/decision_log.py [--subset all_nolex+np]
Writes results/<subset>/decisions@clean.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala import benchmarks as bm, evaluate as ev  # noqa: E402
from pravaala.embed import Space  # noqa: E402
from pravaala.normalise import canon_nasal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all_nolex+np")
ap.add_argument("--evalsan", type=Path, default=ROOT / "data" / "evalsan_clean")
ap.add_argument("--tag", default="clean")
a = ap.parse_args()
RES = ROOT / "results"
dev = bm.load_relatedness(a.evalsan, "dev")
mcq_dev = bm.load_synonym_mcq(a.evalsan)[0::2]


def load(subset, run, center=False, respell=False):
    s = Space.load(RES / subset / run / "vectors.npz")
    if center:
        s = Space(s.words, s.X - s.X.mean(0), s.counts)
    profile = subset.split("+", 1)[1] if "+" in subset else ""
    s.canon = canon_nasal if "n" in profile else None
    if respell:
        s.use_respelling()
    return s


def compare(options):
    """options: {label: Space}. Coverage per option; quality on common support."""
    spaces = list(options.values())
    rel_c = [x for x in dev if all(x[0] in s and x[1] in s for s in spaces)]
    mcq_c = [x for x in mcq_dev if all(x[0] in s and all(o in s for o in x[1]) for s in spaces)]
    out = {}
    for k, s in options.items():
        r_own, _ = ev.relatedness(s, dev)
        m_own = ev.synonym_mcq(s, mcq_dev)
        r, _ = ev.relatedness(s, rel_c)
        m = ev.synonym_mcq(s, mcq_c)
        out[k] = {"rel_cov": r_own["coverage"], "mcq_cov": m_own["coverage"],
                  "rho": r["spearman"], "auc20": r["auc_2v0"], "mcq": m["acc"],
                  "n_rel": r["n"], "n_mcq": m["n"]}
        print(f"  {k:28s} " + " ".join(f"{x}={v:.3f}" if isinstance(v, float) else f"{x}={v}"
                                        for x, v in out[k].items()), flush=True)
    return out


D = {}
S = a.subset
print("D1 training corpus (lexicons)")
D["lexicons"] = {m: compare({"with lexicons": load("all", f"{m}_lem_d300_w5_m5_s0"),
                             "without lexicons": load("all_nolex", f"{m}_lem_d300_w5_m5_s0")})
                 for m in ("ppmi_svd", "sgns")}
print("D2 normalisation profile")
D["profile"] = {m: compare({p or "none": load(f"all_nolex{'+' + p if p else ''}", f"{m}_lem_d300_w5_m5_s0")
                            for p in ("", "n", "np", "npc")})
                for m in ("ppmi_svd", "sgns")}
print("D3 view")
D["view"] = {m: compare({v: load(S, f"{m}_{v}_d100_w5_m5_s0") for v in ("seg", "lem", "lid")})
             for m in ("ppmi_svd", "sgns", "fasttext", "glove_ref")}
print("D6 mean-centring")
D["centring"] = {m: compare({"raw": load(S, f"{m}_lem_d100_w5_m5_s0"),
                             "centred": load(S, f"{m}_lem_d100_w5_m5_s0", center=True)})
                 for m in ("ppmi_svd", "sgns", "fasttext", "glove_ref")}
print("D9 respelling")
D["respelling"] = {m: compare({"exact lookup": load(S, f"{m}_lem_d100_w5_m5_s0"),
                               "+ respelling": load(S, f"{m}_lem_d100_w5_m5_s0", respell=True)})
                   for m in ("ppmi_svd", "sgns")}

# ---- decisions from dev-only sweeps ---------------------------------------
sw = json.load(open(RES / S / f"sweep_ppmi_lem@{a.tag}.json"))
pick = lambda w, d, p: next(r for r in sw["sweep"] if (r["window"], r["dim"], r["p"]) == (w, d, p))
sel = sw["selected"]
D["ppmi_hyper"] = {"n_configs": len(sw["sweep"]), "grid": {"window": sorted({r["window"] for r in sw["sweep"]}),
                            "dim": sorted({r["dim"] for r in sw["sweep"]}),
                            "p": sorted({r["p"] for r in sw["sweep"]})},
                   "default (w5, d100, p=.5)": pick(5, 100, 0.5),
                   f"selected (w{sel['window']}, d{sel['dim']}, p={sel['p']})": pick(sel["window"], sel["dim"], sel["p"])}
imp = json.load(open(RES / S / f"improve@{a.tag}.json"))
D["technique"] = {k: {"dev_rho": v["dev_rho"], "dev_mcq": v["dev_mcq_even"]}
                  for k, v in imp.items() if not k.startswith("_") and "dev_rho" in v}
D["technique"]["_selected"] = imp.get("_selected")
sm = json.load(open(RES / S / f"sweep_smoothing_lem@{a.tag}.json"))["rows"]
D["smoothing"] = {r["variant"]: {k: r["dev"][k] for k in ("rho", "auc20", "mcq", "base_rho", "base_mcq")}
                  for r in sm if r["min_count"] == 5}
D["min_count"] = {r["min_count"]: {k: r["dev"][k] for k in ("rel_cov", "mcq_cov", "rho", "mcq", "base_rho", "base_mcq")}
                  for r in sm if r["variant"] == "a75"}
out = RES / S / f"decisions@{a.tag}.json"
json.dump(D, open(out, "w"), indent=1, default=float)
print("written", out)
