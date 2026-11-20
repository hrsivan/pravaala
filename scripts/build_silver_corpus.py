"""Turn ByT5-Sanskrit annotations of GRETIL (from notebooks/silver_annotate.ipynb)
into corpus views, with exactly the gold data's normalisation.

Input : data/silver/gretil/units.tsv            (from prepare_gretil.py)
        data/silver/gretil/annotated/**/shard_XXXX.tsv  (idx \\t raw model output;
        any number of runs, e.g. one folder per Kaggle version; repeats are ignored)
Output: data/corpora/gretil+<profile>/           silver only
        data/corpora/<base>_gretil+<profile>/    gold <base> + silver
        views seg.txt and lem.txt (no lid/sense: silver lemmas have no IDs),
        meta.tsv (provenance), stats.json.

Each model output token is unsandhied_lemma_TAG (IAST); TAG is mapped to UD
features with the analyser's own table (data/silver/sanskrit_tags.tsv), and
the word then goes through dcs.norm_word, the function the gold DCS uses.
Sentences whose output is malformed (no parsable token, or fewer analysed
words than half the input's words) are dropped and counted.

Both output subsets are ordinary corpus directories, so train.py, evaluate.py
and learning_curve.py work on them unchanged; deleting them, data/silver/ and
this script leaves no trace in the rest of the project.

usage: python scripts/build_silver_corpus.py --base all_nolex --profile np
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala.dcs import norm_word  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--silver", type=Path, default=ROOT / "data" / "silver" / "gretil")
ap.add_argument("--tags", type=Path, default=ROOT / "data" / "silver" / "sanskrit_tags.tsv")
ap.add_argument("--base", default="all_nolex", help="gold subset to combine with")
ap.add_argument("--profile", default="np")
ap.add_argument("--corpora", type=Path, default=ROOT / "data" / "corpora")
a = ap.parse_args()
if "c" in a.profile:
    sys.exit("profile flag 'c' (compound lexicon) is not supported for silver data")

TAGS = dict(l.rstrip("\n").split("\t")[:2] for l in open(a.tags, encoding="utf-8") if "\t" in l)
units = [l.rstrip("\n").split("\t") for l in open(a.silver / "units.tsv", encoding="utf-8")][1:]
shard_lines = int(json.load(open(a.silver / "stats.json"))["config"]["shard_lines"])


def parse(raw: str):
    words = []
    for item in raw.split():
        parts = item.split("_")
        if len(parts) != 3 or not parts[0] or not parts[1]:
            continue
        unsandhied, lemma, tag = parts
        words.append(norm_word(unsandhied.rstrip("-"), lemma, TAGS.get(tag, ""), a.profile))
    return [(s, l) for s, l in words if s and l]


suffix = f"+{a.profile}" if a.profile else ""
silver_dir = a.corpora / f"gretil{suffix}"
silver_dir.mkdir(parents=True, exist_ok=True)
fs = {v: open(silver_dir / f"{v}.txt", "w", encoding="utf-8") for v in ("seg", "lem")}
fm = open(silver_dir / "meta.tsv", "w", encoding="utf-8")
st = Counter()
# outputs may come from several runs (Colab, Kaggle versions): any shard_*.tsv
# under annotated/, each unit taken once, in corpus order
ann = {}
for f in sorted((a.silver / "annotated").rglob("shard_*.tsv")):
    base = int(f.stem.split("_")[1]) * shard_lines
    for line in open(f, encoding="utf-8"):
        idx, _, raw = line.rstrip("\n").partition("\t")
        if idx.isdigit():
            ann.setdefault(base + int(idx), raw)
for g in sorted(ann):
    raw = ann[g]
    uid, file, iast = units[g]
    st["sentences_in"] += 1
    w = parse(raw)
    if not w or len(w) < 0.5 * len(iast.split()):
        st["dropped_malformed"] += 1
        continue
    fs["seg"].write(" ".join(s for s, _ in w) + "\n")
    fs["lem"].write(" ".join(l for _, l in w) + "\n")
    fm.write(f"{uid}\tGRETIL:{file}\t-1\tsilver\n")   # same columns as gold meta.tsv
    st["sentences_kept"] += 1
    st["tokens_kept"] += len(w)
for x in (*fs.values(), fm):
    x.close()
json.dump(dict(st), open(silver_dir / "stats.json", "w"), indent=1)

gold = a.corpora / f"{a.base}{suffix}"
both = a.corpora / f"{a.base}_gretil{suffix}"
both.mkdir(parents=True, exist_ok=True)
for v in ("seg", "lem"):
    with open(both / f"{v}.txt", "w", encoding="utf-8") as out:
        for src in (gold / f"{v}.txt", silver_dir / f"{v}.txt"):
            with open(src, encoding="utf-8") as fin:
                shutil.copyfileobj(fin, out)
gold_n = sum(1 for _ in open(gold / "lem.txt", encoding="utf-8"))
with open(both / "meta.tsv", "w", encoding="utf-8") as out:
    out.write(open(gold / "meta.tsv", encoding="utf-8").read())
    out.write(open(silver_dir / "meta.tsv", encoding="utf-8").read())
json.dump({"gold_sentences": gold_n, **{f"silver_{k}": v for k, v in st.items()}},
          open(both / "stats.json", "w"), indent=1)
print(json.dumps(dict(st), indent=1))
