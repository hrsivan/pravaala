#!/usr/bin/env python3
"""
Extract (sandhied chunk -> unsandhied morphs) pairs from a DCS-style CoNLL-U corpus.

How the format encodes this:
  * Multiword-token line "2-3  kumāramabhimantrayate" = the sandhied chunk.
    The following rows (FORM "_") hold the split parts; the clean form is in
    MISC as Unsandhied=...  (e.g. kumāram + abhimantrayate).
  * A plain row with a real FORM is a one-morph chunk (FORM may differ from
    Unsandhied only by light normalisation, e.g. jātaṃ -> jātam).
  * "Sub-sentences" (sent_id like 433853_0) have FORM "_" everywhere and no
    range lines; they are paired with the whitespace words of "# text" when
    the counts match (otherwise they only feed the morph lexicon).

Outputs (in --out dir):
  chunks.tsv            surface<TAB>morph1+morph2+...   (all chunks)
  train.tsv / test.tsv  random 3/4 : 1/4 split of chunks, as in the paper
  unsandhied_words.txt  one morph per line, duplicates kept (for train())
  stats.txt

Usage:
  python extract_unsandhied.py corpus.conllu --out data --slp1
"""
import argparse
import os
import random
import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------- IAST -> SLP1
_MULTI = {
    "kh": "K", "gh": "G", "ch": "C", "jh": "J", "ṭh": "W", "ḍh": "Q",
    "th": "T", "dh": "D", "ph": "P", "bh": "B",
}
_SINGLE = {
    "a": "a", "ā": "A", "i": "i", "ī": "I", "u": "u", "ū": "U",
    "ṛ": "f", "ṝ": "F", "ḷ": "x", "ḹ": "X", "e": "e", "o": "o",
    "ṃ": "M", "ṁ": "M", "ḥ": "H", "ṅ": "N", "ñ": "Y", "ṭ": "w", "ḍ": "q",
    "ṇ": "R", "ś": "S", "ṣ": "z", "k": "k", "g": "g", "c": "c", "j": "j",
    "t": "t", "d": "d", "n": "n", "p": "p", "b": "b", "m": "m", "y": "y",
    "r": "r", "l": "l", "v": "v", "s": "s", "h": "h", "ḻ": "L", "'": "'",
    "’": "'",
}


def iast_to_slp1(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    out, i = [], 0
    while i < len(s):
        if s[i:i + 2] in ("ai", "au"):          # diphthongs
            out.append("E" if s[i:i + 2] == "ai" else "O")
            i += 2
        elif s[i:i + 2] in _MULTI:
            out.append(_MULTI[s[i:i + 2]])
            i += 2
        else:
            out.append(_SINGLE.get(s[i], s[i]))
            i += 1
    return "".join(out)


# ---------------------------------------------------------------- CoNLL-U
def parse_misc(misc: str) -> dict:
    d = {}
    if misc and misc != "_":
        for kv in misc.split("|"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                d[k] = v
    return d


def read_sentences(path):
    sent = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                m = re.match(r"#\s*(\w+)\s*=\s*(.*)", line)
                if m:
                    if m.group(1) == "text":
                        if sent and sent["rows"]:
                            yield sent
                        sent = {"text": m.group(2).strip(), "rows": []}
                    elif sent is not None:
                        sent[m.group(1)] = m.group(2).strip()
                continue
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 10 or sent is None:
                continue
            sent["rows"].append({
                "id": cols[0], "form": cols[1], "lemma": cols[2],
                "misc": parse_misc(cols[9]),
            })
    if sent and sent["rows"]:
        yield sent


def morph_of(row):
    # Unsandhied is the clean form; fall back to FORM if missing.
    return row["misc"].get("Unsandhied") or (row["form"] if row["form"] != "_" else None)


def extract(sent):
    """Return (chunks, loose_morphs). chunks = [(surface, [morphs])]."""
    rows, chunks, orphans = sent["rows"], [], []
    i = 0
    while i < len(rows):
        r = rows[i]
        if "-" in r["id"]:                                   # multiword token
            a, b = map(int, r["id"].split("-"))
            parts = []
            j = i + 1
            while j < len(rows) and "-" not in rows[j]["id"] and a <= int(rows[j]["id"]) <= b:
                m = morph_of(rows[j])
                if m:
                    parts.append(m)
                j += 1
            if parts:
                chunks.append((r["form"], parts))
            i = j
        else:
            m = morph_of(r)
            if m:
                if r["form"] != "_":
                    chunks.append((r["form"], [m]))
                else:
                    orphans.append(m)
            i += 1
    loose = []
    if orphans:
        words = sent["text"].split()
        if not chunks and len(words) == len(orphans):        # sub-sentence, 1:1
            chunks.extend((w, [m]) for w, m in zip(words, orphans))
        else:
            loose = orphans
    return chunks, loose


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", help="one or more .conllu files")
    ap.add_argument("--out", default="data")
    ap.add_argument("--slp1", action="store_true", help="transliterate IAST -> SLP1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.75)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    conv = iast_to_slp1 if args.slp1 else (lambda x: x)

    all_chunks, morph_tokens = [], []
    for path in args.corpus:
        for sent in read_sentences(path):
            chunks, loose = extract(sent)
            for surf, morphs in chunks:
                all_chunks.append((conv(surf), [conv(m) for m in morphs]))
                morph_tokens.extend(conv(m) for m in morphs)
            morph_tokens.extend(conv(m) for m in loose)

    def dump(name, items):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for s, ms in items:
                f.write(f"{s}\t{'+'.join(ms)}\n")

    dump("chunks.tsv", all_chunks)
    rng = random.Random(args.seed)
    shuffled = all_chunks[:]
    rng.shuffle(shuffled)
    k = int(len(shuffled) * args.train_frac)
    dump("train.tsv", shuffled[:k])
    dump("test.tsv", shuffled[k:])
    with open(os.path.join(args.out, "unsandhied_words.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(morph_tokens) + "\n")

    split_chunks = sum(1 for _, ms in all_chunks if len(ms) > 1)
    stats = (f"chunks: {len(all_chunks)}\nmulti-morph chunks: {split_chunks}\n"
             f"morph tokens: {len(morph_tokens)}\nmorph types: {len(set(morph_tokens))}\n"
             f"phoneme types: {len(Counter(''.join(morph_tokens)))}\n")
    with open(os.path.join(args.out, "stats.txt"), "w") as f:
        f.write(stats)
    print(stats)


if __name__ == "__main__":
    main()