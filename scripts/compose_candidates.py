"""OPTIONAL PLUGIN step 1: propose segmentations for absent benchmark words.

usage: python scripts/compose_candidates.py --subset all \
           --corpora data/corpora --evalsan EVALSAN/Data --amara amara_pada.csv
writes data/compose/<subset>/candidates.tsv for human review. Set `status`
to `approved` / `rejected` (or edit `split`), then pass
`--plugins compose` to run_experiment.py.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vidruma import benchmarks as bm  # noqa: E402
from vidruma.plugins import compose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="all")
ap.add_argument("--corpora", type=Path, default=ROOT / "data" / "corpora")
ap.add_argument("--evalsan", type=Path, required=True)
ap.add_argument("--amara", type=Path, required=True)
ap.add_argument("--min_count", type=int, default=5)
ap.add_argument("--margin_bits", type=float, default=4.0)
ap.add_argument("--out", type=Path, default=ROOT / "data" / "compose")
a = ap.parse_args()

lem = Counter()
for line in open(a.corpora / a.subset / "lem.txt", encoding="utf-8"):
    lem.update(line.split())
words = {w for x, y, _ in bm.load_relatedness(a.evalsan, "test") for w in (x, y)}
words |= {w for x, y, _ in bm.load_relatedness(a.evalsan, "dev") for w in (x, y)}
words |= {w for q, o in bm.load_synonym_mcq(a.evalsan) for w in [q, *o]}
words |= {e.word for e in bm.load_amarakosa(a.amara)}
absent = {w for w in words if lem.get(w, 0) < a.min_count}

seg = compose.Segmenter(lem, min_count=a.min_count)
rows = compose.candidates(absent, seg, a.margin_bits)
out = a.out / a.subset
out.mkdir(parents=True, exist_ok=True)
compose.write_tsv(rows, out / "candidates.tsv")
st = Counter(r["status"] for r in rows)
print(f"{len(absent)} absent/rare words -> {dict(st)}  ->  {out / 'candidates.tsv'}")
