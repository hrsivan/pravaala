r"""OOV accounting across preprocessing views (RQ3).

Let B be a benchmark word set (stems, SLP1 after normalisation), and for view
k and frequency threshold tau let  V_k(tau) = { types t : #_k(t) >= tau }.
Membership of a stem b in V_k is  b in strip(V_k(tau))  for the ID views.
Define  OOV_k(tau) = |B \ V_k(tau)| / |B|.

The total OOV of the naive pipeline (raw benchmark encoding vs. surface
tokens) is decomposed telescopically along the chain of normalisations:

  OOV_raw,surf = [OOV_raw,surf - OOV_surf]        (encoding mismatch)
               + [OOV_surf - OOV_seg]             (sandhi)
               + [OOV_seg  - OOV_lem]             (inflection)
               + [OOV_lem  - OOV_lid]             (homograph split; <= 0)
               +  OOV_lid                         (lexical sparsity)

and the last term splits further into  absent (#=0)  vs  rare (0<#<tau).
A telescoping decomposition depends on the order of the chain; here the
order is forced (lemmas are only defined on segmented text), so the
attribution is canonical. Each term is a difference of proportions on the
same B, so paired (McNemar-type) tests apply to each step.
"""
from __future__ import annotations

from collections import Counter

from .dcs import strip_id


def string_counts(view_counts: Counter, id_view: bool) -> Counter:
    """For ID views, the count relevant for membership of a string s is the
    max over IDs realising s (a homograph is in-vocab if *some* sense is)."""
    if not id_view:
        return view_counts
    out = Counter()
    for t, c in view_counts.items():
        s = strip_id(t)
        out[s] = max(out[s], c)
    return out


def coverage(B: set[str], counts: Counter, tau: int) -> dict:
    present = {b for b in B if counts.get(b, 0) >= tau}
    rare = {b for b in B if 0 < counts.get(b, 0) < tau}
    return {"n": len(B), "in_vocab": len(present) / len(B),
            "rare": len(rare) / len(B),
            "absent": 1 - (len(present) + len(rare)) / len(B)}


def decompose(B_raw: set[str], B: set[str], counts_by_view: dict[str, Counter],
              tau: int) -> dict:
    """Telescoping decomposition (see module docstring). counts_by_view must
    contain string-level counts for surf, seg, lem, lid."""
    oov = lambda S, v: 1 - coverage(S, counts_by_view[v], tau)["in_vocab"]
    o_raw = oov(B_raw, "surf")
    o = {v: oov(B, v) for v in ("surf", "seg", "lem", "lid")}
    cov_lid = coverage(B, counts_by_view["lid"], tau)
    return {
        "total_naive": o_raw,
        "encoding": o_raw - o["surf"],
        "sandhi": o["surf"] - o["seg"],
        "inflection": o["seg"] - o["lem"],
        "homograph_split": o["lem"] - o["lid"],
        "residual_rare": cov_lid["rare"],
        "residual_absent": cov_lid["absent"],
    }
