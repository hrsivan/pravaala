"""Train one embedding model on one view of one corpus subset.

usage: python scripts/train.py --subset all --view lid --model sgns [--dim 300 ...]
writes results/<subset>/<run>/vectors.npz and config.json, where
<run> = <model>_<view>_d<dim>_w<window>_m<min_count>_s<seed>.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala import paths  # noqa: E402
from pravaala import embed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all")
ap.add_argument("--view", default="lid", choices=["surf", "seg", "lem", "lid", "sense"])
ap.add_argument("--model", default="sgns", choices=["sgns", "fasttext", "ppmi_svd", "glove_ref"],
                help="sgns = word2vec skip-gram with negative sampling; "
                     "glove_ref = the reference Stanford GloVe (needs --glove_bin)")
ap.add_argument("--dim", type=int, default=300)
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--min_count", type=int, default=5)
ap.add_argument("--epochs", type=int, default=5, help="word2vec/FastText; GloVe uses --glove_epochs")
ap.add_argument("--glove_epochs", type=int, default=25)
ap.add_argument("--glove_bin", type=Path, default=paths.GLOVE_BIN,
                help="glove_ref: build/ folder of github.com/stanfordnlp/GloVe")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corpora", type=Path, default=ROOT / "data" / "corpora")
ap.add_argument("--results", type=Path, default=ROOT / "results")
ap.add_argument("--evalsan", type=Path, help="FastText only: precompute subword vectors")
ap.add_argument("--amara", type=Path, help="FastText only: precompute subword vectors")
a = ap.parse_args()

run = f"{a.model}_{a.view}_d{a.dim}_w{a.window}_m{a.min_count}_s{a.seed}"
out = a.results / a.subset / run
out.mkdir(parents=True, exist_ok=True)
corpus = a.corpora / a.subset / f"{a.view}.txt"
t0 = time.time()
if a.model in ("sgns", "fasttext"):
    kv = embed.train_sgns(corpus, a.dim, a.window, a.min_count, epochs=a.epochs,
                          workers=a.workers, seed=a.seed, fasttext=a.model == "fasttext")
    space = embed.Space.from_kv(kv)
    if a.model == "fasttext" and a.evalsan and a.amara:
        # subword vectors for benchmark strings unseen in training; stored
        # separately so evaluation can include or exclude them (`ours+subword`)
        from pravaala import benchmarks as bm
        import numpy as np
        B = {w for x, y, _ in bm.load_relatedness(a.evalsan, "test") for w in (x, y)}
        B |= {w for x, y, _ in bm.load_relatedness(a.evalsan, "dev") for w in (x, y)}
        B |= {w for q, o in bm.load_synonym_mcq(a.evalsan) for w in [q, *o]}
        B |= {w for ws in bm.load_categorisation(a.evalsan, "sem").values() for w in ws}
        B |= {e.word for e in bm.load_amarakosa(a.amara)}
        oov = sorted(w for w in B if w not in space)
        np.savez_compressed(out / "subword_oov.npz", words=np.array(oov),
                            X=np.stack([kv[w] for w in oov]).astype(np.float32))
elif a.model == "glove_ref":
    from pravaala.glove import train_glove_reference
    words, X, counts = train_glove_reference(corpus, a.glove_bin, a.dim, a.window, a.min_count,
                                             epochs=a.glove_epochs, threads=a.workers,
                                             seed=a.seed, workdir=out / "glove_work")
    import shutil
    shutil.rmtree(out / "glove_work", ignore_errors=True)
    space = embed.Space(words, X, counts)
else:
    sents = lambda: (l.split() for l in open(corpus, encoding="utf-8"))
    vocab, counts = embed.build_vocab(sents, a.min_count)
    C = embed.cooccurrence(sents, vocab, a.window)
    X = embed.svd_embeddings(C, a.dim, seed=a.seed)            # ppmi_svd
    space = embed.Space(list(vocab), X, counts)
space.save(out / "vectors.npz")
cfg = {k: str(v) for k, v in vars(a).items()} | {"run": run, "vocab": len(space.words),
                                                  "train_seconds": round(time.time() - t0, 1)}
json.dump(cfg, open(out / "config.json", "w"), indent=1)
print(json.dumps(cfg, indent=1))
