# Silver annotation on Kaggle

Notebook: `notebooks/silver_annotate_kaggle.ipynb` (two T4 GPUs, one model copy each).
Input: `pravaala_silver_shards.zip` (91 shards, 1.8M units of GRETIL, from `scripts/prepare_gretil.py`).

## One-time setup

1. **Verify your phone number** (kaggle.com → profile → Settings → Phone verification).
   Kaggle requires it for GPUs and for internet access in notebooks; the notebook needs
   internet to download the model from Hugging Face.
2. **Upload the shards as a private dataset**: kaggle.com → Datasets → New Dataset →
   drop in `pravaala_silver_shards.zip` → title `pravaala-silver-shards` → keep it Private → Create.
   Kaggle unzips it; the notebook finds the shards wherever they end up.
3. **Import the notebook**: kaggle.com → Code → New Notebook → File → Import Notebook →
   upload `silver_annotate_kaggle.ipynb`.
4. In the notebook's right-hand panel:
   * **Add Input** → Your Work → Datasets → `pravaala-silver-shards`;
   * Session options → **Accelerator: GPU T4 x2**;
   * Session options → **Internet: On**.

## Each run (about 11 hours, unattended)

5. *(First time, optional but recommended)* run the first four cells interactively, up to
   and including the fp16 check. It prints units/s and the estimated total GPU-session hours.
   Send that line to Claude. Then **stop the interactive session** (it uses quota too).
6. **Save Version** (top right) → **Save & Run All (Commit)** → Save. You can close the
   browser; the run continues on Kaggle's side and stops by itself after `RUN_HOURS` (11 h).
7. When the version finishes: open it (Code → your notebook → Versions) → **Output** →
   download `annotated.zip` and `progress.json`. Keep the latest `annotated.zip` safe.

## The next runs

8. Before the next Save & Run All, attach the previous run's output so its work is skipped:
   **Add Input** → Your Work → Notebooks → this notebook (its latest version's output).
   If Kaggle does not offer the notebook itself as an input, use either:
   * the version's Output tab → **New Dataset** (creates a dataset from the output), then
     attach that dataset; or
   * upload the downloaded `annotated.zip` as a new version of a private dataset
     `pravaala-silver-annotated` and attach that.
   The notebook picks up any `shard_*.tsv` under its inputs, so all three routes work, and
   repeated units are skipped. Each new version's output then holds everything so far.
9. Repeat 6–8. The weekly GPU quota (about 30 h, shown in your profile settings, resetting
   weekly) allows two such runs per week. `progress.json` shows the share done.

## When done (or when we decide to stop early)

10. Send Claude the latest `annotated.zip`. It goes to `data/silver/gretil/annotated/`, then
    `python scripts/build_silver_corpus.py --base all_nolex --profile np` builds the corpora.
    Stopping early is fine: shards are processed in a fixed genre-mixed order.

## Settings at the top of the notebook

* `RUN_HOURS = 11.0`: when to stop starting new batches (Kaggle ends sessions at 12 h; the
  remaining time covers the model download, the fp16 check and zipping the output).
* `BATCH = 128`: units per batch per GPU; halved automatically on out-of-memory.
* `FORCE_FP32 = False`: the notebook uses fp16 only if it gives the same output as fp32 on
  at least 99% of 256 sample units; set True to always use fp32.
