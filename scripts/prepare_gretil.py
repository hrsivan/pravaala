"""Prepare GRETIL Sanskrit text as input for silver annotation (ByT5-Sanskrit).

Pipeline (each step's drops are counted in stats.json):
  1. extract  : strip HTML, keep lines that look like IAST Sanskrit (no English,
                no apparatus), remove line references, split at dandas and
                cap units at --max_bytes (the analyser's output is ~3x longer
                than its input, and generation is capped at 512 bytes);
  2. lexicons : skip the lexicography section (6_sastra/2_lex) and the
                lexicons filed elsewhere (--lex_files), as in the no-lexicon
                DCS subset;
  3. quotes   : drop units sharing >= --quote_frac of their character 12-grams
                with any lexicon (GRETIL 2_lex + the nine DCS lexicons):
                commentaries quote kosa verses, which would reintroduce the leak;
  4. DCS      : drop units sharing >= --dcs_frac of their 12-grams with the
                DCS (already gold-annotated; the analyser was trained on it);
  5. dedup    : drop exact repeats (same skeleton) within GRETIL.
12-grams are taken over a *skeleton* that ignores spacing and the variation
editions differ in (nasals, visarga/sibilant before consonants, avagraha).

Output data/silver/gretil/: units.tsv (uid, file, iast), shards/*.txt for
the notebook (one unit per line, same order as units.tsv), stats.json,
dropped_sample.tsv.

usage: python scripts/prepare_gretil.py --gretil <mirror>/gretil.sub.uni-goettingen.de/gretil/1_sanskr
"""
import argparse
import html
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala.translit import iast_to_slp1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gretil", type=Path, required=True)
ap.add_argument("--out", type=Path, default=ROOT / "data" / "silver" / "gretil")
ap.add_argument("--dcs_all", type=Path, default=ROOT / "data" / "corpora" / "all",
                help="DCS corpus dir with surf.txt and meta.tsv (full DCS, lexicons included)")
ap.add_argument("--lex_dir", default="6_sastra/2_lex")
ap.add_argument("--lex_files", nargs="*",
                default=["4_rellit/saiva/bijnighu.htm", "4_rellit/saiva/purekaku.htm",
                         "6_sastra/7_ayur/rajnighu.htm", "6_sastra/7_ayur/vanighau.htm",
                         "6_sastra/7_ayur/vanighiu.htm", "6_sastra/7_ayur/vanighpu.htm"],
                help="lexicons filed outside --lex_dir (found by title: Bija-, Raja-, "
                     "Astanganighantu, Ekaksarakosa); excluded and used for the quote filter")
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--dcs_frac", type=float, default=0.5)
ap.add_argument("--quote_frac", type=float, default=0.3)
ap.add_argument("--max_bytes", type=int, default=160)
ap.add_argument("--shard_lines", type=int, default=20000)
a = ap.parse_args()
cfg = json.load(open(ROOT / "config" / "subsets.json"))
LEX_TEXTS = set(cfg["all_nolex"]["exclude_texts"])

# ------------------------------------------------------------------ text
IAST = set("abcdeghijklmnoprstuvyāīūṛṝḷḹṃḥṅñṭḍṇśṣ '-")
ENGLISH = re.compile(r"\b(the|and|of|with|for|this|that|from|by|is|are|was|in|on|to|as|at|or|be|it|"
                     r"text|edition|ed|input|version|long|short|vocalic|vowel|consonant|sequence|"
                     r"multibyte|description|unicode|utf|font|file|copyright|usage|reference|"
                     r"purposes|converted|browser|revised|based|entered|data|page|vol|see|cf|"
                     r"anusvara|visarga|avagraha|retroflex|palatal|velar|dental|labial|nasal|"
                     r"aspirate|sibilant|guttural|transliteration|gretil|manuscript|ms|mss)\b")
REF = re.compile(r"\S*\d\S*")                       # line numbers, verse references, sigla
DANDA = re.compile(r"\s*(?:\|\||//|\||/|।।|॥|।)\s*")
TAG = re.compile(r"<[^>]+>")


def sanskrit_lines(raw: str):
    text = unicodedata.normalize("NFC", html.unescape(TAG.sub("\n", raw)))
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("%") or "{" in line:
            continue
        line = REF.sub(" ", line.replace("ṁ", "ṃ")).lower()
        line = re.sub(r"[\[\](){}<>*+=_\"“”‘’:;,.?!]", " ", line)
        if ENGLISH.search(line):
            continue
        for part in DANDA.split(line):
            part = " ".join(part.split()).strip(" -")
            if len(part.split()) < 2 or any(ch not in IAST for ch in part):
                continue
            if not any(ch in "āīūṛṃḥṅñṭḍṇśṣ" for ch in part) and len(part) > 40:
                continue                             # long diacritic-free stretch: not IAST
            yield part


def cap(unit: str, max_bytes: int):
    """Split a unit at word boundaries into pieces of at most max_bytes (UTF-8)."""
    out, cur = [], []
    for w in unit.split():
        if cur and len(" ".join(cur + [w]).encode()) > max_bytes:
            out.append(" ".join(cur)); cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


SKEL_MAP = str.maketrans({"M": "m", "N": "m", "Y": "m", "R": "m", "n": "m",
                          "H": "s", "S": "s", "z": "s", "'": None})
NONLETTER = re.compile(r"[^A-Za-z]")


def skeleton(slp1: str) -> bytes:
    return NONLETTER.sub("", slp1.translate(SKEL_MAP)).encode("ascii", "ignore")


def kgram_hashes(b: bytes, k: int) -> np.ndarray:
    x = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
    n = len(x) - k + 1
    if n <= 0:
        return np.zeros(0, np.uint64)
    h = np.zeros(n, np.uint64)
    P = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * P + x[j:j + n]
    return h


def shingle_set(skels, k):
    arrs = [kgram_hashes(s, k) for s in skels]
    return np.unique(np.concatenate(arrs)) if arrs else np.zeros(0, np.uint64)


def overlap(h: np.ndarray, ref: np.ndarray) -> float:
    if len(h) == 0 or len(ref) == 0:
        return 0.0
    i = np.searchsorted(ref, h).clip(0, len(ref) - 1)
    return float(np.mean(ref[i] == h))


# ------------------------------------------------------- reference sets
print("reference shingles ...", flush=True)
dcs_skels, dcslex_skels = [], []
for sl, meta in zip(open(a.dcs_all / "surf.txt", encoding="utf-8"),
                    open(a.dcs_all / "meta.tsv", encoding="utf-8")):
    s = skeleton(sl)
    dcs_skels.append(s)
    if meta.split("\t")[1] in LEX_TEXTS:
        dcslex_skels.append(s)
DCS = shingle_set(dcs_skels, a.k)
DCS_LINES = set(dcs_skels)
lex_files = sorted((a.gretil / a.lex_dir).rglob("*.htm")) + [a.gretil / f for f in a.lex_files]
gretil_lex = [skeleton(iast_to_slp1(u)) for f in lex_files
              for u in sanskrit_lines(f.read_bytes().decode("utf-8", "replace"))]
LEX = shingle_set(gretil_lex + dcslex_skels, a.k)
print(f"  DCS 12-grams {len(DCS):,}; lexicon 12-grams {len(LEX):,} "
      f"({len(lex_files)} GRETIL lexicon files + {len(dcslex_skels)} DCS lexicon sentences)", flush=True)

# --------------------------------------------------------------- main
files = sorted(p for p in a.gretil.rglob("*.htm") if a.lex_dir not in p.as_posix()
               and p.relative_to(a.gretil).as_posix() not in a.lex_files)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "shards").mkdir(exist_ok=True)
stats = Counter()
tok = Counter()
seen = set()
dropped = []
units_f = open(a.out / "units.tsv", "w", encoding="utf-8")
units_f.write("uid\tfile\tiast\n")
shard, shard_n, shard_f = 0, 0, None
uid = 0
per_coll = Counter()
for fi, p in enumerate(files):
    rel = p.relative_to(a.gretil).as_posix()
    raw = p.read_bytes().decode("utf-8", errors="replace")
    for unit in sanskrit_lines(raw):
        for piece in cap(unit, a.max_bytes):
            stats["units_seen"] += 1
            n = len(piece.split())
            tok["seen"] += n
            sk = skeleton(iast_to_slp1(piece))
            h = kgram_hashes(sk, a.k)
            reason = None
            if sk in seen:
                reason = "duplicate"
            elif overlap(h, LEX) >= a.quote_frac:
                reason = "lexicon_quote"
            elif sk in DCS_LINES or overlap(h, DCS) >= a.dcs_frac:
                reason = "in_dcs"
            seen.add(sk)
            if reason:
                stats[reason] += 1
                tok[reason] += n
                if random.random() < 0.002:
                    dropped.append((reason, rel, piece))
                continue
            stats["kept"] += 1
            tok["kept"] += n
            per_coll[rel.split("/")[0]] += n
            if shard_f is None or shard_n >= a.shard_lines:
                if shard_f:
                    shard_f.close()
                shard_f = open(a.out / "shards" / f"shard_{shard:04d}.txt", "w", encoding="utf-8")
                shard, shard_n = shard + 1, 0
            units_f.write(f"g{uid}\t{rel}\t{piece}\n")
            shard_f.write(piece + "\n")
            uid += 1
            shard_n += 1
    if fi % 500 == 0:
        print(f"  {fi}/{len(files)} files, kept {tok['kept']:,} tokens", flush=True)
units_f.close()
if shard_f:
    shard_f.close()
out = {"files": len(files), "lexicon_files_skipped": len(lex_files), "units": dict(stats),
       "tokens": dict(tok), "kept_tokens_by_collection": dict(per_coll), "shards": shard,
       "config": {k: str(v) for k, v in vars(a).items()}}
json.dump(out, open(a.out / "stats.json", "w"), indent=1)
with open(a.out / "dropped_sample.tsv", "w", encoding="utf-8") as f:
    for r in dropped:
        f.write("\t".join(r) + "\n")
print(json.dumps(out, indent=1))
