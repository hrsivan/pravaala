"""Generate the LaTeX tables (and inline numbers, facts.json) of the
mid-submission report from result files. Every number in the report's tables
comes from here.

usage: python report/midsem/make_midsem_tables.py

  main.tex       4 models x 3 views, test, cleaned EvalSan, leak-free corpus
                 (all_nolex+np), respelling on; centring as chosen on dev
                 (word2vec and FastText centred, PPMI-SVD and GloVe raw)
  rq1.tex        the same runs on their common support, with exact McNemar
                 tests (seg vs lem, lem vs lid), Holm-corrected
  compare.tex    Sandhan et al.'s published numbers vs ours under their own
                 protocol and original benchmark files
  decisions.tex  the dev-set decision log (scripts/decision_log.py)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import benchmarks as bm  # noqa: E402
from vidruma.evaluate import auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all_nolex+np")
ap.add_argument("--tag", default="clean")
ap.add_argument("--evalsan", type=Path, default=ROOT / "data" / "evalsan_clean")
a = ap.parse_args()
OUT = Path(__file__).parent / "tables"
OUT.mkdir(exist_ok=True)
RES = ROOT / "results" / a.subset
# model key, display name, protocol chosen on dev (centring)
MODELS = [("sgns", "word2vec", "ours+center"), ("fasttext", "FastText", "ours+center"),
          ("glove_ref", "GloVe", "ours"), ("ppmi_svd", "PPMI-SVD", "ours")]
VIEWS = ["seg", "lem", "lid"]
TAG = f"+spell@{a.tag}"
f = lambda x, p=3: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"
pct = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"
run = lambda m, v: f"{m}_{v}_d100_w5_m5_s0"
facts = {}


def table(path, colspec, head, rows):
    (OUT / path).write_text("\n".join([rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *head,
                                       r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]) + "\n")


def holm(ps):
    order = np.argsort(ps)
    adj, run_max = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run_max
    return adj


# ------------------------------------------------------------------ main grid
rows = []
for m, M, proto in MODELS:
    for i, v in enumerate(VIEWS):
        R = json.load(open(RES / run(m, v) / f"eval_{proto}{TAG}.json"))
        r, s, k, c = R["relatedness_test"], R["synonym_mcq"], R["ak_varga_kmeans"], R.get("cat_sem", {})
        rows.append(" & ".join([M if i == 0 else "", rf"\vk{{{v}}}", pct(r["coverage"]),
                                f(r["spearman"]), f(r["auc_2v0"]), f(r["auc_1v0"]), f(r["macroF1_3way"]),
                                pct(s["coverage"]), pct(s["acc"]), f(c.get("nmi")), f(k["nmi"])]) + r" \\")
        facts[f"{m}_{v}"] = {"rho": r["spearman"], "auc20": r["auc_2v0"], "auc10": r["auc_1v0"],
                             "f1": r["macroF1_3way"], "rel_cov": r["coverage"], "mcq": s["acc"],
                             "mcq_cov": s["coverage"], "cat_nmi": c.get("nmi"), "ak_nmi": k["nmi"],
                             "ak_null": k["nmi_null"]}
    rows.append(r"\midrule")
table("main.tex", "llrrrrrrrrr",
      [r" & & \multicolumn{5}{c}{Relatedness} & \multicolumn{2}{c}{Synonyms} & Cat. & Varga\\",
       r"\cmidrule(lr){3-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}\cmidrule(lr){11-11}",
       r"model & view & cov & $\rho$ & AUC$_{20}$ & AUC$_{10}$ & F1 & cov & acc & NMI & NMI\\"],
      rows[:-1])

# --------------------------------------------- RQ1 on common support, McNemar
y = np.array([t for *_, t in bm.load_relatedness(a.evalsan, "test")])
I = {(m, v): json.load(open(RES / run(m, v) / f"items_{p}{TAG}.json")) for m, _, p in MODELS for v in VIEWS}
rel = sorted(set.intersection(*(set(d["relatedness_test"]) for d in I.values())), key=int)
syn = sorted(set.intersection(*(set(d["synonym_mcq"]) for d in I.values())), key=int)
yy = y[[int(i) for i in rel]]
facts["common_rel_n"], facts["common_syn_n"] = len(rel), len(syn)


def mcnemar(d1, d2):
    b = sum(d1[i] > d2[i] for i in syn)
    c = sum(d1[i] < d2[i] for i in syn)
    return b, c, (binomtest(b, b + c).pvalue if b + c else 1.0)


tests, cells = [], {}
for m, M, _ in MODELS:
    for v in VIEWS:
        s = np.array([I[(m, v)]["relatedness_test"][i] for i in rel])
        cells[(m, v)] = (spearmanr(s, yy).statistic,
                         float(np.mean([I[(m, v)]["synonym_mcq"][i] for i in syn])))
    for v1, v2 in (("seg", "lem"), ("lem", "lid")):
        tests.append((m, v1, v2, *mcnemar(I[(m, v2)]["synonym_mcq"], I[(m, v1)]["synonym_mcq"])))
adj = holm(np.array([t[-1] for t in tests]))
pv = {(t[0], t[1], t[2]): (t[3], t[4], t[5], q) for t, q in zip(tests, adj)}
facts["mcnemar"] = {f"{m}:{v1}->{v2}": {"b": b, "c": c, "p": p, "p_holm": q}
                    for (m, v1, v2), (b, c, p, q) in pv.items()}
rows = []
for m, M, _ in MODELS:
    sl, ll = pv[(m, "seg", "lem")], pv[(m, "lem", "lid")]
    sig = lambda q: f(q, 3) if q >= 0.001 else "$<$.001"
    rows.append(" & ".join([M, *(f(cells[(m, v)][0], 2) for v in VIEWS),
                            *(pct(cells[(m, v)][1]) for v in VIEWS), sig(sl[3]), f(ll[3], 2)]) + r" \\")
table("rq1.tex", "lrrrrrrrr",
      [r" & \multicolumn{3}{c}{$\rho$} & \multicolumn{3}{c}{Synonym acc.} & \multicolumn{2}{c}{$p_{\mathrm{Holm}}$}\\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
       r" & \vk{seg} & \vk{lem} & \vk{lid} & \vk{seg} & \vk{lem} & \vk{lid} & s$\to$l & l$\to$i\\"],
      rows)

# -------------------------------------------------- comparison with Sandhan et al.
# Published: arXiv:2104.00270, Table 3 (relatedness macro-F1 %, semantic and
# syntactic categorisation purity, synonym accuracy %).
PUB = {"word2vec": (30.90, 0.41, 0.22, 37.34), "GloVe": (31.50, 0.39, 0.25, 37.41),
       "FastText": (31.20, 0.47, 0.14, 37.87), "ELMo": (37.17, 0.41, 0.86, 42.15)}
ours = {}
for m, M, _ in MODELS:
    E = json.load(open(RES / run(m, "lem") / "eval_evalsan.json"))
    Es = json.load(open(RES / run(m, "seg") / "eval_evalsan.json"))
    ours[M] = (100 * E["relatedness_test"]["macroF1"], E["cat_sem"]["purity_best_of_6"],
               Es["cat_syn"]["purity_best_of_6"], 100 * E["synonym_mcq"]["accuracy"],
               E["synonym_mcq"]["n"])
sub = json.load(open(RES / run("fasttext", "lem") / "eval_ours+subword.json"))["synonym_mcq"]
facts["fasttext_subword_mcq"] = {"acc": sub["acc"], "n": sub["n"]}
facts["evalsan_protocol"] = ours
rows = [r"\multicolumn{6}{l}{\textit{Sandhan et al.\ (published)}}\\"]
rows += [" & ".join([k, f(v[0], 1), f(v[1], 2), f(v[2], 2), f(v[3], 1), "all"]) + r" \\"
         for k, v in PUB.items()]
rows += [r"\midrule", r"\multicolumn{6}{l}{\textit{Ours (leak-free corpus)}}\\"]
rows += [" & ".join([k, f(v[0], 1), f(v[1], 2), f(v[2], 2), f(v[3], 1), f"{v[4]}"]) + r" \\"
         for k, v in ours.items()]
rows += [" & ".join(["FastText + subw.", "", "", "", f(100 * sub["acc"], 1), f"{sub['n']}"]) + r" \\"]
table("compare.tex", "lrrrrr",
      [r" & Rel. & \multicolumn{2}{c}{Cat.\ purity} & \multicolumn{2}{c}{Synonyms}\\",
       r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
       r"model & F1 & sem & syn & acc & items\\"], rows)

# ------------------------------------------------------------- decision log
D = json.load(open(RES / f"decisions@{a.tag}.json"))
P = "ppmi_svd"
lx = D["lexicons"][P]
pr = D["profile"][P]
vw = D["view"][P]
ce = D["centring"]
sm, mc = D["smoothing"], D["min_count"]
hy = D["ppmi_hyper"]
hk = [k for k in hy if k.startswith("selected")][0]
tq = {k: v for k, v in D["technique"].items() if not k.startswith("_")}
base_t = tq.get("PPMI-SVD baseline (k=1, p=0.5)", {})
best_t = max((v for k, v in tq.items() if k.startswith("PPMI") and "baseline" not in k),
             key=lambda v: v["dev_rho"] + v["dev_mcq"], default={"dev_rho": None, "dev_mcq": None})
rs = D["respelling"][P]
w2v5 = next(v for k, v in tq.items() if k.startswith("SGNS 5"))
w2v15 = next(v for k, v in tq.items() if k.startswith("SGNS 15"))
rq = lambda d: f"{d['rho']:.3f}/{100 * d['mcq']:.1f}"
rows = [
    rf"Lexicons in training & with {rq(lx['with lexicons'])}; without {rq(lx['without lexicons'])} & remove (leak; the drop is its size)\\",
    rf"Normalisation profile & coverage {pct(pr['none']['rel_cov'])}/{pct(pr['n']['rel_cov'])}/{pct(pr['np']['rel_cov'])}/{pct(pr['npc']['rel_cov'])}\% (none/\vk{{n}}/\vk{{np}}/\vk{{npc}}); $\rho$ {f(pr['none']['rho'])}--{f(pr['npc']['rho'])} & \vk{{np}}\\",
    rf"View & \vk{{seg}} {rq(vw['seg'])}; \vk{{lem}} {rq(vw['lem'])}; \vk{{lid}} {rq(vw['lid'])} & \vk{{lem}} primary\\",
    rf"PPMI-SVD settings & $w5,d100$: {f(hy['default (w5, d100, p=.5)']['dev_rho'])}/{pct(hy['default (w5, d100, p=.5)']['dev_mcq'])}; best of {hy['n_configs']}: {f(hy[hk]['dev_rho'])}/{pct(hy[hk]['dev_mcq'])} & $w10,d300,p{{=}}0.5$\\",
    rf"Shift, $w{{+}}c$, ABTT, dep.\ contexts & baseline {f(base_t.get('dev_rho'))}/{pct(base_t.get('dev_mcq'))}; best variant {f(best_t['dev_rho'])}/{pct(best_t['dev_mcq'])} & none (no gain)\\",
    rf"word2vec training & 5 epochs, 5 negatives: {f(w2v5['dev_rho'])}/{pct(w2v5['dev_mcq'])}; 15 epochs, 10 negatives: {f(w2v15['dev_rho'])}/{pct(w2v15['dev_mcq'])} & 15/10 in the final grid\\",
    rf"Mean-centring & $\Delta\rho$: PPMI {ce['ppmi_svd']['centred']['rho'] - ce['ppmi_svd']['raw']['rho']:+.3f}, w2v {ce['sgns']['centred']['rho'] - ce['sgns']['raw']['rho']:+.3f}, FT {ce['fasttext']['centred']['rho'] - ce['fasttext']['raw']['rho']:+.3f}, GloVe {ce['glove_ref']['centred']['rho'] - ce['glove_ref']['raw']['rho']:+.3f} & centre w2v, FastText\\",
    rf"PMI smoothing & none {f(sm['a75']['rho'])}/{pct(sm['a75']['mcq'])}; discount {f(sm['a75+D']['rho'])}/{pct(sm['a75+D']['mcq'])}; KN cont.\ {f(sm['cont']['rho'])}/{pct(sm['cont']['mcq'])} & none\\",
    rf"Minimum count & coverage {pct(mc['5']['rel_cov'])}/{pct(mc['3']['rel_cov'])}/{pct(mc['2']['rel_cov'])}\% (5/3/2); $\rho$ on old items {f(mc['5']['base_rho'])}/{f(mc['3']['base_rho'])}/{f(mc['2']['base_rho'])} & 5 (revisit with silver data)\\",
    rf"Respelling at lookup & coverage {pct(rs['exact lookup']['rel_cov'])}$\to${pct(rs['+ respelling']['rel_cov'])}\%; quality unchanged & adopt\\",
]
table("decisions.tex", "p{2.9cm}p{9.0cm}p{2.7cm}",
      [r"Decision & Dev evidence (PPMI-SVD, \vk{lem}; $\rho$/synonym acc.\ on common support) & Choice\\"],
      rows)

# ---------------------------------------------------------------- facts
lc = json.load(open(RES / f"learning_curve@{a.tag}.json"))
facts["learning_curve"] = lc
facts["decisions"] = D
json.dump(facts, open(OUT / "facts.json", "w"), indent=1, default=float)
print("tables:", sorted(p.name for p in OUT.glob("*.tex")))
