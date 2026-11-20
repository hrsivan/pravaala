r"""RQ2: which vargas does the vector space recover, and why?

usage: python scripts/analyse_vargas.py results/<subset>/<run> --amara CSV \
           [--corpora data/corpora] [--groups config/varga_groups.json]

Per varga g (gold class G = g, cluster label K from the run's k-means items):
  n_g          words evaluated
  F1_g         best-match cluster F-measure  max_k 2|c_k ∩ g| / (|c_k| + |g|)
  z_g          (F1_g - null mean) / null sd under 200 label shuffles; F1 is
               strongly size-biased (rho(F1, n_g) ~ 0.8), z removes that
  disp_g       dispersion  H(K | G = g) / log2 min(n_g, k)   in [0, 1]
               (0 = all words in one cluster, 1 = maximally scattered)
  H_category   taxonomic entropy H(category | G = g) from the Amarakosa's own
               ontological category column (Miller-Madow corrected)
  Htext_res    mean residual (on log2 freq) of H(text | W = w): how spread a
               word is over DCS texts; low = topically concentrated (prakarana)
  Hctx_g       mean context entropy H(C | W = w) of its words (lem view, ±5)
  Hres_g       mean residual of Hctx after regressing on log2 frequency
               (plug-in entropy grows with sample size, so raw Hctx is
               confounded with frequency; the residual removes the linear part)
Group comparison (concrete / abstract / cosmological from the config):
  permutation test on the difference of mean lift between two groups,
  vargas as units (all relabelings of the 18 vargas, or 10^5 random ones).
Writes <run>/analysis_vargas.json.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala import benchmarks as bm  # noqa: E402
from pravaala import embed  # noqa: E402
from pravaala.infotheory import entropy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--amara", type=Path, required=True)
ap.add_argument("--corpora", type=Path, default=ROOT / "data" / "corpora")
ap.add_argument("--groups", type=Path, default=ROOT / "config" / "varga_groups.json")
ap.add_argument("--protocol", default="ours")
ap.add_argument("--n_perm", type=int, default=100_000)
ap.add_argument("--measure", default="spec_info_bits", choices=["spec_info_bits", "lift", "z"])
a = ap.parse_args()

subset = a.run.parent.name
_prof = subset.split("+", 1)[1] if "+" in subset else ""
if "n" in _prof:
    from pravaala.normalise import canon_nasal as _canon
else:
    _canon = lambda w: w
lab = json.load(open(a.run / f"items_{a.protocol}.json"))["ak_varga_kmeans"]
ak_entries = bm.load_amarakosa(a.amara)
w2v, _ = bm.ak_partitions(ak_entries)
# Amarakosa's own ontological category per word (most frequent over entries)
_cats = defaultdict(Counter)
for e in ak_entries:
    if e.category:
        _cats[e.word][e.category] += 1
cat_of = {w: c.most_common(1)[0][0] for w, c in _cats.items()}
words = sorted(lab)
gold = {w: next(iter(w2v[w])) for w in words}
k = len(set(lab.values()))

# --- context entropy on the lem view -------------------------------------
corpus = a.corpora / subset / "lem.txt"
sents = lambda: (l.split() for l in open(corpus, encoding="utf-8"))
vocab, counts = embed.build_vocab(sents, 5)
C = embed.cooccurrence(sents, vocab, window=5, harmonic=False).tocsr()
hctx, logf = {}, {}
for w in words:
    if _canon(w) in vocab:
        i = vocab[_canon(w)]
        hctx[w] = entropy(C.data[C.indptr[i]:C.indptr[i + 1]])
        logf[w] = np.log2(counts[i])
# text (genre) distribution of each word: H(text | W = w); topical
# concentration is Bhartrhari's prakarana, a document-level context
texts = [l.split("\t")[1] for l in open(a.corpora / subset / "meta.tsv", encoding="utf-8")]
wanted = set(words)
canon_of = {_canon(w): w for w in words}
wanted_c = set(canon_of)
tcount = defaultdict(Counter)
for line, t in zip(open(corpus, encoding="utf-8"), texts):
    for tok in line.split():
        if tok in wanted_c:
            tcount[canon_of[tok]][t] += 1
htext = {w: entropy(tcount[w].values()) for w in words if w in tcount}
ws = [w for w in words if w in hctx]
x, y = np.array([logf[w] for w in ws]), np.array([hctx[w] for w in ws])
beta = np.polyfit(x, y, 1)
resid = dict(zip(ws, y - np.polyval(beta, x)))
yt = np.array([htext[w] for w in ws])
beta_t = np.polyfit(x, yt, 1)
resid_t = dict(zip(ws, yt - np.polyval(beta_t, x)))

# --- per-varga statistics --------------------------------------------------
by_g = defaultdict(list)
for w in words:
    by_g[gold[w]].append(w)
size_k = Counter(lab.values())


def best_f1(mem_labels, n_g):
    kc = Counter(mem_labels)
    return max(2 * c / (size_k[kk] + n_g) for kk, c in kc.items())


# per-varga permutation null: shuffle gold labels over words (cluster sizes
# and varga sizes fixed); F1 is size-biased, so we report lift = F1 - null
# mean and z = lift / null sd, which compare across vargas of different size.
rng0 = np.random.default_rng(1)
gold_arr = np.array([gold[w] for w in words])
lab_arr = np.array([lab[w] for w in words])
null, null_h = defaultdict(list), defaultdict(list)
for _ in range(200):
    gp = rng0.permutation(gold_arr)
    for g in by_g:
        sel = lab_arr[gp == g]
        null[g].append(best_f1(sel.tolist(), len(sel)))
        null_h[g].append(entropy(Counter(sel.tolist()).values()))
rows = {}
for g, mem in by_g.items():
    kc = Counter(lab[w] for w in mem)
    f1 = best_f1([lab[w] for w in mem], len(mem))
    denom = np.log2(min(len(mem), k)) if len(mem) > 1 else 1.0
    mu, sd = float(np.mean(null[g])), float(np.std(null[g]) + 1e-12)
    h_g = entropy(kc.values())
    # taxonomic entropy of the varga (Miller-Madow corrected): how
    # heterogeneous its referents are by the Amarakosa's own categories
    cc = Counter(cat_of[w] for w in mem if w in cat_of)
    rows[g] = {"n": len(mem),
               # specific information i_g = H(K) - H(K|G=g); I(K;G) = sum_g P(g) i_g.
               # Plug-in H(K|G=g) is biased low for small n_g, so we report the
               # bias-corrected version  E_null[H(K|G=g)] - H(K|G=g)  (bits).
               "spec_info_bits": float(np.mean(null_h[g]) - h_g),
               "Htext_res": float(np.mean([resid_t[w] for w in mem if w in resid_t])),
               "H_category": entropy(cc.values(), miller_madow=True) if cc else float("nan"),
               "n_categories": len(cc),
               "F1": f1, "F1_null": mu, "lift": f1 - mu,
               "z": (f1 - mu) / sd,
               "dispersion": entropy(kc.values()) / denom,
               "Hctx": float(np.mean([hctx[w] for w in mem if w in hctx])),
               "Hres": float(np.mean([resid[w] for w in mem if w in resid])),
               "log2freq": float(np.mean([logf[w] for w in mem if w in logf]))}

groups = {k_: v for k_, v in json.load(open(a.groups)).items() if not k_.startswith("_")}
gof = {g: name for name, vs in groups.items() for g in vs}
for g in rows:
    rows[g]["group"] = gof.get(g, "?")

V = sorted(rows)
# effect size for comparisons: lift = F1 - null mean (z is a significance
# measure and grows with n_g, so it must not be compared across vargas)
F = np.array([rows[g][a.measure] for g in V])
out = {"run": a.run.name, "k": k, "n_words": len(words), "measure": a.measure,
       "ctx_entropy_vs_logfreq_slope": float(beta[0]),
       "per_varga": rows,
       "spearman_measure_vs_Hres": float(spearmanr(F, [rows[g]["Hres"] for g in V]).statistic),
       "spearman_measure_vs_Hctx": float(spearmanr(F, [rows[g]["Hctx"] for g in V]).statistic),
       "spearman_measure_vs_n": float(spearmanr(F, [rows[g]["n"] for g in V]).statistic),
       "spearman_measure_vs_Htext_res": float(spearmanr(F, [rows[g]["Htext_res"] for g in V]).statistic),
       "spearman_measure_vs_Htext_res_p": float(spearmanr(F, [rows[g]["Htext_res"] for g in V]).pvalue),
       "spearman_measure_vs_Hcategory": float(spearmanr(F, [rows[g]["H_category"] for g in V]).statistic),
       "spearman_measure_vs_Hcategory_p": float(spearmanr(F, [rows[g]["H_category"] for g in V]).pvalue),
       "group_means": {}}
lab_g = np.array([rows[g]["group"] for g in V])
for name in groups:
    sel = lab_g == name
    out["group_means"][name] = {"vargas": int(sel.sum()), "measure": float(F[sel].mean()),
                                "Hres": float(np.mean([rows[g]["Hres"] for g in np.array(V)[sel]]))}

# exact / Monte-Carlo permutation test: concrete vs abstract mean F1
rng = np.random.default_rng(0)
idx = np.where(np.isin(lab_g, ["concrete", "abstract"]))[0]
Fi, Li = F[idx], lab_g[idx]
obs = Fi[Li == "concrete"].mean() - Fi[Li == "abstract"].mean()
nc = int((Li == "concrete").sum())
perm = np.array([(lambda p: Fi[p[:nc]].mean() - Fi[p[nc:]].mean())(rng.permutation(len(Fi)))
                 for _ in range(a.n_perm)])
out["concrete_minus_abstract"] = float(obs)
out["perm_p_one_sided"] = float(((perm >= obs).sum() + 1) / (a.n_perm + 1))

json.dump(out, open(a.run / "analysis_vargas.json", "w"), indent=1)
print(f"{'varga':22s} {'grp':5s} {'n':>5s} {'F1':>6s} {'null':>6s} {'lift':>6s} {'z':>6s} {'Ibits':>6s} {'Hcat':>5s} {'Htx':>5s} {'disp':>6s} {'Hctx':>6s} {'Hres':>6s}")
for g in sorted(V, key=lambda g: -rows[g][a.measure]):
    r = rows[g]
    print(f"{g:22s} {r['group'][:5]:5s} {r['n']:5d} {r['F1']:6.3f} {r['F1_null']:6.3f} {r['lift']:6.3f} {r['z']:6.1f} {r['spec_info_bits']:6.3f} {r['H_category']:5.2f} {r['Htext_res']:+5.2f} {r['dispersion']:6.3f} "
          f"{r['Hctx']:6.2f} {r['Hres']:6.2f}")
print(json.dumps({k_: v for k_, v in out.items() if k_ != "per_varga"}, indent=1))
