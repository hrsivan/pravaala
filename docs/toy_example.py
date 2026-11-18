"""Every number in docs/FROM_SCRATCH.md's toy example, computed.

usage: python docs/toy_example.py     (no dependencies beyond numpy, scipy, sklearn)
Run it alongside the tutorial: each printed block corresponds to a section.
"""
import itertools
import math
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

np.set_printoptions(precision=3, suppress=True)
log2 = math.log2


def H(counts):
    """Entropy (bits) of the distribution given by a list of counts."""
    n = sum(counts)
    return -sum(c / n * log2(c / n) for c in counts if c)


# ----------------------------------------------------------------- 1. corpus
# (surface, segmented, lemmatised) for each sentence, all in SLP1.
CORPUS = [
    ("jalaM pibati",   "jalam pibati",      "jala pA"),
    ("toyaM pibati",   "toyam pibati",      "toya pA"),
    ("nadyAM jalam",   "nadyAm jalam",      "nadI jala"),
    ("nadyAM toyam",   "nadyAm toyam",      "nadI toya"),
    ("agnirdahati",    "agniH dahati",      "agni dah"),
    ("vahnirdahati",   "vahniH dahati",     "vahni dah"),
    ("dIpe'gniH",      "dIpe agniH",        "dIpa agni"),
    ("dIpe vahniH",    "dIpe vahniH",       "dIpa vahni"),
    ("jalamagniSca",   "jalam agniH ca",    "jala agni ca"),
    ("toyaM vahniSca", "toyam vahniH ca",   "toya vahni ca"),
]
print("== 1. views: token and type counts")
for i, name in enumerate(["surf", "seg", "lem"]):
    toks = [t for s in CORPUS for t in s[i].split()]
    print(f"{name:5s} tokens={len(toks):2d} types={len(set(toks)):2d}")

# ------------------------------------------------ 1b. OOV in the toy benchmark
print("\n== 1b. why benchmark words are missing (tau = 1)")
WX2SLP1 = str.maketrans({"t": "w", "T": "W", "d": "q", "D": "Q", "N": "R", "w": "t", "W": "T",
                         "x": "d", "X": "D", "f": "N", "F": "Y", "R": "z", "q": "f", "Q": "F"})
bench_wx = ["jala", "woya", "naxI", "agni", "vahni", "xIpa", "salila"]   # toy benchmark, WX-coded
bench = [w.translate(WX2SLP1) for w in bench_wx]
for name, i in [("surf", 0), ("seg", 1), ("lem", 2)]:
    types = {t for sent in CORPUS for t in sent[i].split()}
    print(f"{name:5s} raw-WX found {sum(w in types for w in bench_wx)}/7   "
          f"converted found {sum(w in types for w in bench)}/7")
print("converted:", bench)

# --------------------------------------------- 2. conditional entropy example
print("\n== 2. conditional entropy (separate mini-sample of 10 occurrences)")
occ = [("Bavati", "BU")] * 3 + [("Bavati", "Bavat")] * 1 + [("jalam", "jala")] * 4 + [("jalena", "jala")] * 2
seg = [s for s, _ in occ]
lem = [l for _, l in occ]
H_seg, H_lem, H_joint = H(Counter(seg).values()), H(Counter(lem).values()), H(Counter(occ).values())
print(f"H(seg)={H_seg:.3f}  H(lem)={H_lem:.3f}  H(seg,lem)={H_joint:.3f}")
print(f"H(lem|seg) = H(seg,lem) - H(seg) = {H_joint - H_seg:.3f}")
print(f"H(seg|lem) = H(seg,lem) - H(lem) = {H_joint - H_lem:.3f}")
print(f"check: 0.4 * H(3/4,1/4) = {0.4 * H([3, 1]):.3f};  0.6 * H(2/3,1/3) = {0.6 * H([2, 1]):.3f}")
lid = [("brahman", "brahman_n")] * 3 + [("brahman", "brahman_m")] * 1 + [("deva", "deva_1")] * 4
print(f"H(lid|lem) on brahman/deva sample = {H(Counter(lid).values()) - H(Counter(l for l, _ in lid).values()):.3f}")

# ------------------------------------------------------ 3. co-occurrence
print("\n== 3. sentence-level co-occurrence counts (lem view)")
sents = [s[2].split() for s in CORPUS]
vocab = ["jala", "toya", "nadI", "pA", "agni", "vahni", "dah", "dIpa", "ca"]
idx = {w: i for i, w in enumerate(vocab)}
C = np.zeros((9, 9))
for s in sents:
    for a, b in itertools.permutations(range(len(s)), 2):   # ordered pairs, distinct positions
        C[idx[s[a]], idx[s[b]]] += 1
print("     " + " ".join(f"{w:>5s}" for w in vocab))
for w in vocab:
    print(f"{w:>5s}" + " ".join(f"{int(x):5d}" for x in C[idx[w]]))
N = C.sum()
row = C.sum(1)
print(f"N = {int(N)}; row sums = {dict(zip(vocab, row.astype(int)))}")

# ------------------------------------------------------------- 4. PMI
print("\n== 4. PMI and PPMI (bits)")
with np.errstate(divide="ignore"):
    PMI = np.log2((C / N) / np.outer(row / N, row / N))
PPMI = np.where(PMI > 0, PMI, 0.0)
for a, b in [("jala", "pA"), ("jala", "nadI"), ("jala", "agni"), ("jala", "ca"), ("agni", "dah")]:
    i, j = idx[a], idx[b]
    print(f"PMI({a},{b}) = log2( ({int(C[i, j])}/{int(N)}) / (({int(row[i])}/{int(N)})*({int(row[j])}/{int(N)})) ) = {PMI[i, j]:.3f}")
print("PPMI rows:")
for w in vocab:
    print(f"{w:>5s}", PPMI[idx[w]])

# KL view: expected PMI of a word's contexts = KL(P(.|w) || P(.))
print("\nE_{c|w}[PMI(w,c)] = KL( P(c|w) || P(c) ):")
for w in ["jala", "ca"]:
    i = idx[w]
    pcw = C[i] / C[i].sum()
    pc = row / N
    kl = sum(p * log2(p / q) for p, q in zip(pcw, pc) if p > 0)
    print(f"  {w:5s} KL = {kl:.3f} bits")


# ---------------------------------------------------------- 5. similarity
def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


print("\n== 5. cosine similarity: raw counts vs PPMI rows")
for a, b in [("jala", "toya"), ("agni", "vahni"), ("jala", "nadI"), ("jala", "agni"), ("toya", "dIpa"), ("jala", "ca")]:
    print(f"cos({a},{b}): counts={cos(C[idx[a]], C[idx[b]]):.3f}  PPMI={cos(PPMI[idx[a]], PPMI[idx[b]]):.3f}")

# ------------------------------------------------------------- 6. SVD
print("\n== 6. truncated SVD of PPMI, d=2, p=0.5")
U, S, Vt = np.linalg.svd(PPMI)
print("singular values:", S)
W = U[:, :2] * np.sqrt(S[:2])
for w in vocab:
    print(f"{w:>5s}", W[idx[w]])
print(f"cos2D(jala,toya)={cos(W[idx['jala']], W[idx['toya']]):.3f}  cos2D(jala,agni)={cos(W[idx['jala']], W[idx['agni']]):.3f}"
      f"  cos2D(jala,dah)={cos(W[idx['jala']], W[idx['dah']]):.3f}   <- d=2 is too small: unrelated groups merge")
W4 = U[:, :4] * np.sqrt(S[:4])
print("d=4: " + "  ".join(f"cos({a},{b})={cos(W4[idx[a]], W4[idx[b]]):.3f}"
                          for a, b in [("jala", "toya"), ("jala", "agni"), ("jala", "dah"), ("jala", "nadI")]))

# -------------------------------------------------------- 7. evaluation
print("\n== 7. evaluation on a toy benchmark (vectors = PPMI rows)")
vec = {w: PPMI[idx[w]] for w in vocab}
pairs = [("jala", "toya", 2), ("agni", "vahni", 2), ("jala", "nadI", 1), ("agni", "dIpa", 1),
         ("jala", "agni", 0), ("toya", "dIpa", 0), ("salila", "jala", 2)]
cov = [(a, b, y) for a, b, y in pairs if a in vec and b in vec]
print(f"coverage = {len(cov)}/{len(pairs)}  (salila is OOV)")
s = np.array([cos(vec[a], vec[b]) for a, b, _ in cov])
y = np.array([t for *_, t in cov])
for (a, b, t), v in zip(cov, s):
    print(f"  {a:5s} {b:5s} gold={t} cos={v:.3f}")
print(f"Spearman rho = {spearmanr(s, y).statistic:.3f}")
pos, neg = s[y == 2], s[y == 0]
auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
print(f"AUC(2 vs 0) = {auc:.3f}")
pos1 = s[y == 1]
auc10 = np.mean([(p > q) + 0.5 * (p == q) for p in pos1 for q in neg])
print(f"AUC(1 vs 0) = {auc10:.3f}")
q, opts = "jala", ["toya", "nadI", "agni", "dIpa"]
sc = [cos(vec[q], vec[o]) for o in opts]
print(f"MCQ q={q}: " + ", ".join(f"{o}={v:.3f}" for o, v in zip(opts, sc)) + f"  -> picks {opts[int(np.argmax(sc))]}")

print("\n== 8. clustering vs gold vargas (k=2)")
words = ["jala", "toya", "nadI", "agni", "vahni", "dIpa"]
gold = ["water", "water", "water", "fire", "fire", "fire"]
from sklearn.cluster import KMeans
Xw = np.array([PPMI[idx[w]] / np.linalg.norm(PPMI[idx[w]]) for w in words])
clusters = KMeans(2, n_init=10, random_state=0).fit_predict(Xw).tolist()
print("clusters:", dict(zip(words, clusters)))
pur = sum(max(Counter(g for g, c in zip(gold, clusters) if c == k).values()) for k in set(clusters)) / len(words)
HK, HG = H(Counter(clusters).values()), H(Counter(gold).values())
HKG = H(Counter(zip(clusters, gold)).values())
I = HK + HG - HKG
print(f"purity={pur:.3f}  H(K)={HK:.3f} H(G)={HG:.3f} H(K,G)={HKG:.3f}  I(K;G)={I:.3f}  "
      f"NMI={2 * I / (HK + HG):.3f}  VI={2 * HKG - HK - HG:.3f}")
