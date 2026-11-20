import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_silver_conversion_matches_gold_treatment(tmp_path):
    from pravaala.dcs import norm_word
    silver = tmp_path / "silver"; (silver / "annotated").mkdir(parents=True)
    (silver / "units.tsv").write_text("uid\tfile\tiast\n"
                                      "g0\tx.htm\tagniṃ dīpe gataḥ\n"
                                      "g1\tx.htm\tbroken output here\n", encoding="utf-8")
    (silver / "stats.json").write_text(json.dumps({"config": {"shard_lines": "20000"}}))
    (silver / "annotated" / "shard_0000.tsv").write_text(
        "0\tagnim_agni_SAM dīpe_dīpa_SLM gataḥ_gam_SNPaM\n1\tgarbage\n", encoding="utf-8")
    corp = tmp_path / "corpora"; (corp / "g+np").mkdir(parents=True)
    for v in ("seg", "lem"):
        (corp / "g+np" / f"{v}.txt").write_text("gold line\n", encoding="utf-8")
    (corp / "g+np" / "meta.tsv").write_text("1\tT\t1\t3\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_silver_corpus.py"),
                    "--silver", str(silver), "--base", "g", "--profile", "np",
                    "--corpora", str(corp)], check=True, capture_output=True,
                   env={"PYTHONPATH": str(ROOT / "src")})
    lem = (corp / "gretil+np" / "lem.txt").read_text().split("\n")[0].split()
    # participle gataH -> stem gata (profile p), nasal canonicalised (profile n), as for gold
    assert lem == [norm_word("agnim", "agni", "", "np")[1], "dIpa",
                   norm_word("gataḥ", "gam", "Case=Nom|Gender=Masc|Number=Sing|VerbForm=Part", "np")[1]]
    assert lem[2] == "gata"
    both = (corp / "g_gretil+np" / "lem.txt").read_text().split("\n")
    assert both[0] == "gold line" and len([l for l in both if l]) == 2
    st = json.loads((corp / "gretil+np" / "stats.json").read_text())
    assert st["dropped_malformed"] == 1 and st["sentences_kept"] == 1


def test_silver_merges_runs_without_duplicates(tmp_path):
    silver = tmp_path / "silver"
    for run in ("run1", "run2"):
        (silver / "annotated" / run).mkdir(parents=True)
    (silver / "units.tsv").write_text("uid\tfile\tiast\ng0\tx\tagniṃ dīpe\ng1\tx\tjalaṃ pibati\n",
                                      encoding="utf-8")
    (silver / "stats.json").write_text(json.dumps({"config": {"shard_lines": "20000"}}))
    (silver / "annotated" / "run1" / "shard_0000.tsv").write_text(
        "1\tjalam_jala_SAM pibati_pā_S3PrIn\n0\tagnim_agni_SAM dīpe_dīpa_SLM\n", encoding="utf-8")
    (silver / "annotated" / "run2" / "shard_0000.tsv").write_text(
        "0\tagnim_agni_SAM dīpe_dīpa_SLM\n", encoding="utf-8")
    corp = tmp_path / "corpora"; (corp / "g+np").mkdir(parents=True)
    for v in ("seg", "lem"):
        (corp / "g+np" / f"{v}.txt").write_text("", encoding="utf-8")
    (corp / "g+np" / "meta.tsv").write_text("", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_silver_corpus.py"),
                    "--silver", str(silver), "--base", "g", "--profile", "np",
                    "--corpora", str(corp)], check=True, capture_output=True,
                   env={"PYTHONPATH": str(ROOT / "src")})
    lem = (corp / "gretil+np" / "lem.txt").read_text().strip().split("\n")
    assert lem == ["agni dIpa", "jala pA"]          # corpus order, each unit once
