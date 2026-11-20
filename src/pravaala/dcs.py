"""Reader for the Digital Corpus of Sanskrit (CoNLL-U dump).

Each annotated word w in DCS carries (FORM, LEMMA, LemmaId, Unsandhied,
WordSem?). We derive a chain of *views* V_k : (corpus) -> (token sequences),
each of which differs from its predecessor in exactly one normalisation:

    surf   sandhied space-delimited strings          (orthographic tokens)
    seg    sandhi-resolved inflected word forms       (- sandhi)
    lem    lemma strings                              (- inflection)
    lid    lemma IDs  "lemma_ID"                      (+ homograph split)
    sense  WordSem sense ID if annotated, else lid    (+ sense split)

Formally, let O be the set of word occurrences. Every view is a function
phi_k : O -> Sigma_k^*, and each induces a partition P_k of O (occurrences
with equal value). By construction phi_lem = strip o phi_lid and
phi_lid = strip' o phi_sense, so  P_lem <= P_lid <= P_sense  (refinement
order: each is a coarsening of the next). NOTE: phi_lem is *not* a function
of phi_seg at the type level -- one inflected form may realise several
lemmas (syncretism across paradigms) -- so P_seg and P_lem are incomparable.
The conditional entropies H(lem | seg) and H(lid | lem) over O quantify
exactly this residual ambiguity (see infotheory.conditional_entropy).
All strings are emitted in SLP1.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .translit import iast_to_slp1

VIEWS = ("surf", "seg", "lem", "lid", "sense")
_PUNCT = re.compile(r"[^\w~']+", re.UNICODE)


@dataclass
class Word:
    form: str
    lemma: str
    lemma_id: str
    unsandhied: str
    upos: str
    senses: tuple[str, ...] = ()
    feats: str = ""
    idx: int = 0          # CoNLL-U word index
    head: int = -1        # CoNLL-U HEAD (-1 = unannotated)


@dataclass
class Sentence:
    sent_id: str
    text: str            # DCS text name, e.g. "Mahābhārata"
    chapter_id: str
    slot: int | None = None   # DCS time slot (1 = Vedic ... 5), see subsets
    surface: list[str] = field(default_factory=list)  # sandhied strings
    words: list[Word] = field(default_factory=list)


def _misc(s: str) -> dict[str, str]:
    if s in ("_", ""):
        return {}
    return dict(kv.split("=", 1) for kv in s.split("|") if "=" in kv)


def read_conllu(path: str | os.PathLike) -> Iterator[Sentence]:
    text = chapter = ""
    sent: Sentence | None = None
    covered_until = 0  # last word index covered by a multiword range
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("## text:"):
                text = line.split(":", 1)[1].strip()
            elif line.startswith("## chapter_id:"):
                chapter = line.split(":", 1)[1].strip()
            elif line.startswith("# sent_id"):
                sent = Sentence(line.split("=", 1)[1].strip(), text, chapter)
                covered_until = 0
            elif not line:
                if sent is not None and sent.words:
                    yield sent
                sent = None
            elif sent is not None and line[0].isdigit():
                cols = line.split("\t")
                if len(cols) < 10:
                    continue
                idx, form = cols[0], cols[1]
                if "-" in idx:                       # multiword (sandhi) token
                    a, b = idx.split("-")
                    sent.surface.append(form)
                    covered_until = int(b)
                    continue
                if "." in idx:                       # empty node
                    continue
                m = _misc(cols[9])
                if "LemmaId" not in m:               # punctuation etc.
                    continue
                if int(idx) > covered_until:         # stand-alone token
                    sent.surface.append(form)
                ws = tuple(x for x in m.get("WordSem", "").split(",") if x)
                head = int(cols[6]) if cols[6].isdigit() else -1
                sent.words.append(Word(form, cols[2], m["LemmaId"],
                                       m.get("Unsandhied", form), cols[3], ws, cols[5],
                                       int(idx), head))
    if sent is not None and sent.words:
        yield sent


def load_time_slots(chapter_info_xml: str | os.PathLike) -> dict[str, int | None]:
    """chapter_id -> DCS time slot, from lookup/chapter-info.xml.

    DCS slots (inferred from their texts): 1 Vedic (Samhitas, Brahmanas,
    Sutras); 2 epic / early classical (Mahabharata, Ramayana, Caraka,
    Astadhyayi); 3-5 later classical, Puranic, medieval and commentarial.
    """
    import xml.etree.ElementTree as ET
    out = {}
    for ch in ET.parse(chapter_info_xml).getroot().iter("chapter"):
        s = ch.findtext("dcsTimeSlot")
        out[ch.findtext("chapterId")] = int(s) if s and s.isdigit() else None
    return out


def iter_corpus(root: str | os.PathLike, slots: dict | None = None,
                keep_slots: set | None = None) -> Iterator[Sentence]:
    """Iterate sentences under DCS `conllu/files` in sorted file order.

    If `keep_slots` is given, only chapters whose time slot is in it are
    read (None in keep_slots admits undated chapters).
    """
    for dirpath, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.endswith(".conllu"):
                for s in read_conllu(Path(dirpath) / f):
                    if slots is not None:
                        s.slot = slots.get(s.chapter_id)
                    if keep_slots is None or s.slot in keep_slots:
                        yield s


def _norm(s: str) -> str:
    return _PUNCT.sub("", iast_to_slp1(s))


def norm_word(unsandhied: str, lemma: str, feats: str, profile: str) -> tuple[str, str]:
    """(seg token, lemma token) of one word (IAST in, SLP1 out) under a
    normalisation profile. Shared by gold DCS words and silver-annotated
    words, so both receive exactly the same treatment."""
    from .normalise import canon_nasal, participle_stem
    seg, lem = _norm(unsandhied), _norm(lemma)
    if "p" in profile and ("VerbForm=Part" in feats or "VerbForm=Gdv" in feats):
        lem = participle_stem(seg, feats) or lem
    if "n" in profile:
        seg, lem = canon_nasal(seg), canon_nasal(lem)
    return seg, lem


def _units(sent: Sentence, profile: str):
    """Per-word (seg token, lemma token, id, compound-form stem, is_cpd_member)
    under a normalisation profile (see normalise.py)."""
    out = []
    for w in sent.words:
        seg, lem = norm_word(w.unsandhied, w.lemma, w.feats, profile)
        out.append((seg, lem, w.lemma_id, seg, "Case=Cpd" in w.feats))
    return out


def compound_pairs(sent: Sentence, profile: str = ""):
    """Adjacent (member, next) lemma-token pairs inside compounds."""
    u = _units(sent, profile)
    for i in range(len(u) - 1):
        if u[i][4] and u[i][1] and u[i + 1][1]:
            yield u[i][1], u[i + 1][1]


def view(sent: Sentence, v: str, profile: str = "", lexicon=None) -> list[str]:
    """Token sequence of a sentence under view v (see module docstring).

    profile: normalisation flags from normalise.py ('' = DCS as is). With 'c'
    a CompoundLexicon must be given; fused compounds become single tokens in
    every view (seg, lem, lid, sense), so all views stay aligned.
    """
    if v == "surf":
        toks = [_norm(t) for t in sent.surface]
        if "n" in profile:
            from .normalise import canon_nasal
            toks = [canon_nasal(t) for t in toks]
        return [t for t in toks if t]
    u = _units(sent, profile)
    groups, i = [], 0
    while i < len(u):                        # group words into fused compounds
        j = i
        if "c" in profile and lexicon is not None:
            while j + 1 < len(u) and u[j][4] and lexicon.fuse(u[j][1], u[j + 1][1]):
                j += 1
        groups.append(list(range(i, j + 1)))
        i = j + 1
    from .normalise import join
    toks = []
    for g in groups:
        if len(g) == 1:
            k = g[0]
            seg, lem, lid = u[k][0], u[k][1], f"{u[k][1]}_{u[k][2]}"
            sense_ok = len(sent.words[k].senses) == 1
            sns = lid + (f"@{sent.words[k].senses[0]}" if sense_ok else "")
        else:   # fused compound: member stems + last element (head form / lemma)
            stem_seg = stem_lem = u[g[0]][3]
            for k in g[1:-1]:
                stem_seg = stem_lem = join(stem_seg, u[k][3])
            last = g[-1]
            stem_seg = join(stem_seg, u[last][3] if u[last][4] else u[last][0])
            stem_lem = join(stem_lem, u[last][3] if u[last][4] else u[last][1])
            seg, lem = stem_seg, stem_lem
            lid = f"{lem}_{'-'.join(u[k][2] for k in g)}"
            sns = lid
        toks.append({"seg": seg, "lem": lem, "lid": lid, "sense": sns}[v])
    return [t for t in toks if t and not t.startswith("_")]


def strip_id(tok: str) -> str:
    """Inverse of the lid/sense decoration: 'kfzRa_123@45' -> 'kfzRa'."""
    return tok.split("_", 1)[0]
