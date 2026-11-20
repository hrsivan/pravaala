"""Loaders for EvalSan tasks and the Amarakosa (Hyderabad digitisation).

All words are returned in SLP1. EvalSan's Amarakosa-derived files
(relatedness, synonym MCQs) are encoded in WX while its categorisation and
analogy files (and its released vocabulary) are SLP1; EvalSan applies no
conversion. `scheme="auto"` detects and converts per file; `scheme="raw"`
reproduces the original behaviour for the OOV ablation.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .translit import deva_to_slp1, detect_scheme, wx_to_slp1

# ---------------------------------------------------------------- EvalSan ---
FILES = {
    "rel_dev": "automated_relatedness_AK_dev.csv",
    "rel_test": "automated_relatedness_AK_test.csv",
    "syn_mcq": "final_synonym_MCQs_AK.csv",
    "cat_sem": "final_semantic_categorization.csv",
    "cat_syn": "final_syntactic_categorization.csv",
}


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.reader(fh) if r]


def _converter(rows, scheme):
    if scheme == "raw":
        return lambda w: w
    if scheme == "wx" or (scheme == "auto" and
                          detect_scheme([w for r in rows for w in r])["guess"] == "WX"):
        return wx_to_slp1
    return lambda w: w


def load_relatedness(data_dir, split="test", scheme="auto"):
    """Pairs (u, v, y) with y in {0: unrelated, 1: same varga, 2: same synset}."""
    rows = _read(Path(data_dir) / FILES[f"rel_{split}"])
    cv = _converter(rows, scheme)
    return [(cv(a), cv(b), int(y)) for a, b, y in rows]


def load_synonym_mcq(data_dir, scheme="auto"):
    """Items (q, [a, b, c, d]) whose gold answer is always index 0 (column a)."""
    rows = _read(Path(data_dir) / FILES["syn_mcq"])[1:]
    cv = _converter(rows, scheme)
    return [(cv(r[0]), [cv(x) for x in r[1:5]]) for r in rows]


def load_categorisation(data_dir, kind="sem", scheme="auto"):
    """Dict category -> list of member words. For the semantic file the first
    column is the category name; the syntactic file has no names (row index)."""
    rows = _read(Path(data_dir) / FILES[f"cat_{kind}"])
    cv = _converter(rows, scheme)
    if kind == "sem":
        return {r[0]: [cv(w) for w in r[1:] if w] for r in rows}
    return {str(i): [cv(w) for w in r if w] for i, r in enumerate(rows)}


# -------------------------------------------------------------- Amarakosa ---
@dataclass(frozen=True)
class AKEntry:
    uid: int
    word: str          # SLP1 stem
    address: str       # kanda.varga.sloka.line.word
    linga: str
    varga: str         # SLP1 varga name
    synset: str        # synset key  (varga, artha)
    artha: str         # head-word/gloss of the synset (SLP1)
    category: str      # ontological category column (may be empty)


def load_amarakosa(csv_path) -> list[AKEntry]:
    """Parse aupasana/amara-quiz `amara_pada.csv` (Hyderabad AK data).

    Columns: uid, pada, address, linga, varga, artha, <hypernym chain...>,
    category. A synset is identified with the pair (varga, artha): all words
    glossed by the same artha within one varga.
    """
    out = []
    for r in _read(csv_path):
        if not r[0].isdigit():
            continue
        varga, artha = deva_to_slp1(r[4]), deva_to_slp1(r[5])
        cat = deva_to_slp1(r[14]) if len(r) > 14 else ""
        out.append(AKEntry(int(r[0]), deva_to_slp1(r[1]), r[2],
                           deva_to_slp1(r[3]), varga, f"{varga}|{artha}",
                           artha, cat))
    return out


def ak_partitions(entries):
    """Return (word -> set of vargas, word -> set of synsets).

    A word may belong to several vargas/synsets (polysemy, esp. in the
    nanartha-varga), so these are *relations*, not functions; RQ2 must handle
    that explicitly (see evaluate.purity with multi-label gold).
    """
    w2v, w2s = defaultdict(set), defaultdict(set)
    for e in entries:
        w2v[e.word].add(e.varga)
        w2s[e.word].add(e.synset)
    return w2v, w2s
