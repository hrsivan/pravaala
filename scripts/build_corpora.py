"""Materialise the DCS views as one-sentence-per-line text files.

usage: python scripts/build_corpora.py DCS_CONLLU_DIR OUT_ROOT --subset classical
       (DCS_CONLLU_DIR is .../dcs/data/conllu; output goes to OUT_ROOT/<subset>/)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala.dcs import VIEWS, compound_pairs, iter_corpus, load_time_slots, view  # noqa: E402
from pravaala.normalise import CompoundLexicon  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("conllu", type=Path, help=".../dcs/data/conllu")
ap.add_argument("out_root", type=Path)
ap.add_argument("--subset", default="all")
ap.add_argument("--config", type=Path, default=ROOT / "config" / "subsets.json")
ap.add_argument("--limit", type=int, default=0, help="max sentences (debug)")
ap.add_argument("--profile", default="", help="normalisation flags: n (nasal), p (participle stems), "
                "c (lexicalised compounds); see src/pravaala/normalise.py")
ap.add_argument("--cpd_min_count", type=int, default=10)
ap.add_argument("--cpd_theta", type=float, default=0.5)
a = ap.parse_args()

spec = json.load(open(a.config))[a.subset]
if isinstance(spec, dict):
    keep, exclude = set(spec["slots"]), set(spec.get("exclude_texts", []))
else:
    keep, exclude = set(spec), set()
slots = load_time_slots(a.conllu / "lookup" / "chapter-info.xml")
name = a.subset + (f"+{a.profile}" if a.profile else "")
out = a.out_root / name
lexicon = None
if "c" in a.profile:       # first pass: compound statistics over the same subset
    lexicon = CompoundLexicon(a.cpd_min_count, a.cpd_theta)
    for s in iter_corpus(a.conllu / "files", slots, keep):
        if s.text in exclude:
            continue
        for x, y in compound_pairs(s, a.profile):
            lexicon.add(x, y)
out.mkdir(parents=True, exist_ok=True)
fhs = {v: open(out / f"{v}.txt", "w", encoding="utf-8") for v in VIEWS}
meta = open(out / "meta.tsv", "w", encoding="utf-8")
stats = Counter()
for i, s in enumerate(iter_corpus(a.conllu / "files", slots, keep)):
    if a.limit and i >= a.limit:
        break
    if s.text in exclude:
        stats["excluded_sentences"] += 1
        continue
    for v in VIEWS:
        toks = view(s, v, a.profile, lexicon)
        fhs[v].write(" ".join(toks) + "\n")
        stats[f"tokens_{v}"] += len(toks)
    stats["sentences"] += 1
    stats[f"slot_{s.slot}"] += len(s.words)
    stats["sense_annotated"] += sum(len(w.senses) > 0 for w in s.words)
    meta.write(f"{s.sent_id}\t{s.text}\t{s.chapter_id}\t{s.slot}\n")
for f in fhs.values():
    f.close()
meta.close()
stats["subset"] = a.subset
stats["profile"] = a.profile
if lexicon is not None:
    fused = [(p, c) for p, c in lexicon.pairs.items() if lexicon.fuse(*p)]
    stats["cpd_pair_types"] = len(lexicon.pairs)
    stats["cpd_fused_pair_types"] = len(fused)
    with open(out / "fused_compounds.tsv", "w", encoding="utf-8") as fh:
        fh.write("left\tright\tcount\tnpmi\n")
        for (x, y), c in sorted(fused, key=lambda t: -t[1]):
            fh.write(f"{x}\t{y}\t{c}\t{lexicon.npmi(x, y):.3f}\n")
json.dump(stats, open(out / "stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
