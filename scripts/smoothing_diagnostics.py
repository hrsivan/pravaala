"""Diagnostics for the smoothing sweep (reads the cached m5 co-occurrence).
1. How much of the top of the PPMI matrix is single-co-occurrence cells.
2. Exponent beta in  #distinct partners(c) ~ #(c)^beta  (Kneser-Ney continuation
   vs context-distribution smoothing alpha = 0.75).
3. Share of each word's PPMI row mass carried by single-co-occurrence cells,
   by word frequency band.
usage: python scripts/smoothing_diagnostics.py [--subset all_nolex+np]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala.infotheory import ppmi_matrix  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--subset", default="all_nolex+np")
ap.add_argument("--file", default="lem_w10_m5.npz"); a = ap.parse_args()
z = np.load(ROOT / "results" / a.subset / "cooc_cache" / a.file)
C = sp.csr_matrix((z["cd"], z["ci"], z["cp"]), shape=tuple(z["shape"]))
R = sp.csr_matrix((z["rd"], z["ri"], z["rp"]), shape=tuple(z["shape"]))
M = ppmi_matrix(C, alpha=0.75).tocoo()
r = np.asarray(R[M.row, M.col]).ravel()
out = {"share_cells_r1": float(np.mean(np.isclose(R.data, 1)))}
for q in (0.99, 0.9, 0.5):
    t = np.quantile(M.data, q)
    out[f"share_r1_in_top{round((1-q)*100)}pct_ppmi"] = float(np.mean(np.isclose(r[M.data >= t], 1)))
f = np.asarray(C.sum(0)).ravel(); k = np.diff(sp.csc_matrix(R).indptr)
m = (f > 0) & (k > 0)
out["beta_partners_vs_freq"] = float(np.polyfit(np.log(f[m]), np.log(k[m]), 1)[0])
from pravaala import embed  # noqa: E402
corpus = ROOT / "data" / "corpora" / a.subset / (a.file.split("_")[0] + ".txt")
_, wf = embed.build_vocab(lambda: (l.split() for l in open(corpus, encoding="utf-8")),
                          int(a.file.rsplit("_m", 1)[1].split(".")[0]))   # token counts, vocab order
mass_r1 = np.bincount(M.row, weights=M.data * np.isclose(r, 1), minlength=C.shape[0])
mass = np.bincount(M.row, weights=M.data, minlength=C.shape[0])
bands = {}
for lo, hi in [(5, 20), (20, 100), (100, 1000), (1000, 1e9)]:
    sel = (wf >= lo) & (wf < hi) & (mass > 0)
    bands[f"{lo}-{int(hi) if hi < 1e9 else 'inf'}"] = round(float(np.mean(mass_r1[sel] / mass[sel])), 3)
out["row_ppmi_mass_from_r1_by_freq"] = bands
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / "results" / a.subset / "smoothing_diagnostics.json", "w"), indent=1)
