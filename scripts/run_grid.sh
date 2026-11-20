#!/usr/bin/env bash
# Grid: 4 models x 3 views on one subset, one seed.
# usage: scripts/run_grid.sh EVALSAN_DATA AMARA_CSV [SUBSET] [DIM] [WORKERS] [BENCH_TAG]
#   Without BENCH_TAG: protocols ours, evalsan, evalsan_raw (original EvalSan).
#   With BENCH_TAG (e.g. clean, for data/evalsan_clean): protocols ours, ours+center,
#   results written as eval_<protocol>@<tag>.json, once with exact lookup and once with
#   respelling (eval_<protocol>+spell@<tag>.json, used by the report), plus the
#   per-varga analysis. GloVe is the reference implementation (needs ext/GloVe, see README).
set -euo pipefail
E=$1; A=$2; S=${3:-all}; D=${4:-100}; W=${5:-2}; T=${6:-}
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for view in seg lem lid; do
  for model in sgns ppmi_svd glove_ref fasttext; do
    run=results/$S/${model}_${view}_d${D}_w5_m5_s0
    if [ ! -f "$run/vectors.npz" ]; then
      python scripts/train.py --subset "$S" --view "$view" --model "$model" --dim "$D" \
        --min_count 5 --epochs 5 --glove_epochs 50 --workers "$W" \
        --evalsan "$E" --amara "$A" | tail -2
    fi
    if [ -z "$T" ]; then
      protos="ours evalsan evalsan_raw"; tagarg=""
    else
      protos="ours ours+center"; tagarg="--bench_tag $T"
    fi
    [ "$model" = fasttext ] && protos="$protos ours+subword"
    python scripts/evaluate.py "$run" --evalsan "$E" --amara "$A" --protocols $protos \
      --n_perm 200 $tagarg > "$run/evaluate${T:+_$T}.log" 2>&1
    if [ -n "$T" ]; then
      python scripts/evaluate.py "$run" --evalsan "$E" --amara "$A" --protocols $protos \
        --n_perm 200 $tagarg --respell > "$run/evaluate_${T}_spell.log" 2>&1
    fi
    if [ -n "$T" ]; then
      python scripts/analyse_vargas.py "$run" --amara "$A" --protocol "ours@$T" \
        --n_perm 20000 > "$run/analysis_vargas.log" 2>&1
    fi
    echo "$(date +%T) done $model $view"
  done
done
