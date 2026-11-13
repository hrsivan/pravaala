"""Default locations of external data (overridable by environment variables).

The DCS is tracked in the repository under Proposal/; the other resources are
cloned into ext/ (git-ignored), see README.md.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_p = lambda var, default: Path(os.environ.get(var, default))

DCS_CONLLU = _p("VIDRUMA_DCS", ROOT / "Proposal" / "OliverHellwig sanskrit master dcs-data" / "conllu")
EVALSAN_ORIG = _p("VIDRUMA_EVALSAN", ROOT / "ext" / "EvalSan" / "evaluations" / "Intrinsic" / "Data")
EVALSAN_CLEAN = ROOT / "data" / "evalsan_clean"
AMARA = _p("VIDRUMA_AMARA", ROOT / "ext" / "amara-quiz" / "database" / "amara_pada.csv")
GLOVE_BIN = _p("VIDRUMA_GLOVE", ROOT / "ext" / "GloVe" / "build")
GRETIL = _p("VIDRUMA_GRETIL", ROOT / "ext" / "GRETIL-mirror" / "gretil.sub.uni-goettingen.de" / "gretil" / "1_sanskr")
