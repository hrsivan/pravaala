"""Q3: how many Amarakosa words absent from the lemma vocabulary would each
handling option recover? (Estimation only; nothing here changes the corpora.)

usage: python scripts/absent_options.py [--min_count 5]
Options measured (each independently, then cumulatively):
  N  nasal normalisation  (anusvara vs homorganic nasal: saMDi / sanDi)
  P  participle stems     (kta/ktavatu etc.: DCS lemmatises gataH -> gam; we
                           recover the stem 'gata' from compound members and by
                           stripping a-/A-stem case endings)
  C  compound tokens      (re-join consecutive Case=Cpd members + head lemma
                           with forward vowel sandhi; every contiguous span of
                           >= 2 members counts as a candidate compound)
Writes results/absent_options.json.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pravaala import paths  # noqa: E402
from pravaala import benchmarks as bm  # noqa: E402
from pravaala.translit import iast_to_slp1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--conllu", type=Path, default=paths.DCS_CONLLU / "files")
ap.add_argument("--amara", type=Path, default=paths.AMARA)
ap.add_argument("--min_count", type=int, default=5)
a = ap.parse_args()

nas = lambda s: re.sub(r"[NYRnm](?=[kKgGcCjJwWqQtTdDpPbB])", "M", s)

# forward vowel sandhi at a compound junction (left stem, right stem)
def join(x, y):
    if not x or not y:
        return x + y
    l, r = x[-1], y[0]
    A = "aA"
    if l in A and r in A: return x[:-1] + "A" + y[1:]
    if l in A and r in "iI": return x[:-1] + "e" + y[1:]
    if l in A and r in "uU": return x[:-1] + "o" + y[1:]
    if l in A and r in "eE": return x[:-1] + "E" + y[1:]
    if l in A and r in "oO": return x[:-1] + "O" + y[1:]
    if l in A and r == "f": return x[:-1] + "ar" + y[1:]
    if l in "iI" and r in "iI": return x[:-1] + "I" + y[1:]
    if l in "uU" and r in "uU": return x[:-1] + "U" + y[1:]
    if l in "iI" and r in "aAuUeEoOf": return x[:-1] + "y" + y
    if l in "uU" and r in "aAiIeEoOf": return x[:-1] + "v" + y
    if l == "t" and r in "gGdDbBjJyrvaAiIuUeEoO": return x[:-1] + "d" + y
    if l == "k" and r in "gGdDbBjJyrvaAiIuUeEoO": return x[:-1] + "g" + y
    if x.endswith("as") and r in "gGdDbBjJyrvmnl": return x[:-2] + "o" + y
    return x + y

# a-/A-stem endings (SLP1), longest first; stem = form minus ending + 'a'
A_END = sorted(["aH", "O", "AH", "am", "An", "ena", "AByAm", "EH", "Aya", "eByaH", "At", "asya",
                "ayoH", "AnAm", "e", "ezu", "Ani", "A", "ayA", "ABiH", "AyE", "AByaH", "AyAH",
                "Am", "AyAm", "Asu", "AH", "E"], key=len, reverse=True)
def part_stem(form):
    for e in A_END:
        if form.endswith(e) and len(form) > len(e) + 1:
            return form[:-len(e)] + "a"
    return None

lem, part, comp = Counter(), Counter(), Counter()
for dp, _, fs in os.walk(a.conllu):
    for f in fs:
        if not f.endswith(".conllu"):
            continue
        members = []    # current run of compound members (SLP1 stems)
        for line in open(os.path.join(dp, f), encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if len(c) < 10 or not c[0].isdigit():
                if not line.strip():
                    members = []
                continue
            feats, misc = c[5], dict(kv.split("=", 1) for kv in c[9].split("|") if "=" in kv)
            if "LemmaId" not in misc:
                continue
            L = iast_to_slp1(c[2])
            U = iast_to_slp1(misc.get("Unsandhied", c[1]))
            lem[L] += 1
            is_part = "VerbForm=Part" in feats
            if "Case=Cpd" in feats:
                if is_part:
                    part[U] += 1
                members.append(U)
                continue
            if is_part and "Tense=" not in feats:   # past participles (kta, ktavatu)
                st = part_stem(U)
                if st:
                    part[st] += 1
            head = L if not is_part else (part_stem(U) or L)
            if members:
                seq = members + [head]
                for i in range(len(seq)):
                    acc = seq[i]
                    for j in range(i + 1, len(seq)):
                        acc = join(acc, seq[j])
                        comp[acc] += 1
            members = []

ak = {e.word for e in bm.load_amarakosa(a.amara)}
t = a.min_count
absent = {w for w in ak if lem.get(w, 0) < t}
lemN = Counter()
for k, v in lem.items():
    lemN[nas(k)] += v
recN = {w for w in absent if lemN.get(nas(w), 0) >= t}
recP = {w for w in absent if part.get(w, 0) >= t}
recC = {w for w in absent if comp.get(w, 0) >= t}
recC_any = {w for w in absent if comp.get(w, 0) >= 1}
union = recN | recP | recC
res = {"amarakosa_words": len(ak), "absent_or_rare_lem": len(absent), "tau": t,
       "N_nasal": len(recN), "P_participle_stem": len(recP), "C_compound_token": len(recC),
       "C_compound_token_seen_at_least_once": len(recC_any),
       "union_NPC": len(union), "remaining": len(absent - union),
       "examples_P": sorted(recP)[:15], "examples_C": sorted(recC)[:15],
       "examples_remaining": sorted(absent - union - recC_any)[:25]}
json.dump(res, open(ROOT / "results" / "absent_options.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
