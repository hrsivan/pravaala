# Silver (machine-annotated) data

Optional and removable: nothing outside `data/silver/`, `scripts/prepare_gretil.py`,
`scripts/build_silver_corpus.py`, `scripts/gretil_check.py`, `notebooks/silver_annotate.ipynb`
and the corpus folders `data/corpora/gretil+*` / `*_gretil+*` depends on it.

1. `python scripts/prepare_gretil.py --gretil <GRETIL-mirror>/gretil.sub.uni-goettingen.de/gretil/1_sanskr`
   (GRETIL mirror: https://github.com/INDOLOGY/GRETIL-mirror) writes `gretil/units.tsv`,
   `gretil/shards/`, `gretil/stats.json`.
2. Run `notebooks/silver_annotate.ipynb` on Colab (T4); put its `annotated/` folder here
   as `gretil/annotated/`.
3. `python scripts/build_silver_corpus.py --base all_nolex --profile np`.

`sanskrit_tags.tsv` is the tag table of the ByT5-Sanskrit analysers
(github.com/sebastian-nehrdich/byt5-sanskrit-analyzers; Nehrdich, Hellwig & Keutzer 2024),
released under the Apache licence 2.0 according to the paper.
