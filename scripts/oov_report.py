"""RQ3 preliminary: OOV decomposition + information-theoretic corpus stats.

usage: python scripts/oov_report.py CORPORA_DIR EVALSAN_DATA_DIR AMARA_CSV
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pravaala import benchmarks as bm  # noqa: E402
from pravaala.infotheory import (conditional_entropy, entropy,  # noqa: E402
                                good_turing_unseen)
from pravaala.oov import decompose, string_counts  # noqa: E402

corp, evdir, amara = map(Path, sys.argv[1:4])
VIEWS = ("surf", "seg", "lem", "lid", "sense")

counts, aligned = {}, {v: [] for v in ("seg", "lem", "lid", "sense")}
fh = {v: open(corp / f"{v}.txt", encoding="utf-8") for v in VIEWS}
for lines in zip(*(fh[v] for v in VIEWS)):
    toks = dict(zip(VIEWS, (l.split() for l in lines)))
    for v in VIEWS:
        counts.setdefault(v, Counter()).update(toks[v])
    if len({len(toks[v]) for v in aligned}) == 1:      # 1:1 aligned words
        for v in aligned:
            aligned[v].extend(toks[v])

report = {"corpus": {}}
for v in VIEWS:
    c = counts[v]
    report["corpus"][v] = {
        "tokens": sum(c.values()), "types": len(c),
        "H_unigram_bits": round(entropy(c.values()), 3),
        "good_turing_unseen": round(good_turing_unseen(c), 4)}
ce = lambda x, y: round(conditional_entropy(aligned[x], aligned[y]), 4)
report["ambiguity_bits"] = {
    "H(lem|seg)": ce("seg", "lem"), "H(lid|lem)": ce("lem", "lid"),
    "H(sense|lid)": ce("lid", "sense"), "H(seg|lem)": ce("lem", "seg")}

str_counts = {v: string_counts(counts[v], v in ("lid", "sense")) for v in VIEWS}
benches = {
    "relatedness_test": [w for a, b, _ in bm.load_relatedness(evdir, "test") for w in (a, b)],
    "synonym_mcq": [w for q, opts in bm.load_synonym_mcq(evdir) for w in [q, *opts]],
    "amarakosa_all": [e.word for e in bm.load_amarakosa(amara)],
}
raw = {
    "relatedness_test": [w for a, b, _ in bm.load_relatedness(evdir, "test", "raw") for w in (a, b)],
    "synonym_mcq": [w for q, opts in bm.load_synonym_mcq(evdir, "raw") for w in [q, *opts]],
    "amarakosa_all": [e.word for e in bm.load_amarakosa(amara)],
}
report["oov"] = {}
for name in benches:
    B, Braw = set(benches[name]), set(raw[name])
    report["oov"][name] = {f"tau={t}": {k: round(x, 4) for k, x in
                           decompose(Braw, B, str_counts, t).items()}
                           for t in (1, 5, 10)}
print(json.dumps(report, indent=1, ensure_ascii=False))
json.dump(report, open(corp / "oov_report.json", "w"), indent=1, ensure_ascii=False)
