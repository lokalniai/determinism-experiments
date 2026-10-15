#!/bin/bash
#PBS -N det_analyze
#PBS -l select=1:ncpus=1:mem=8gb:scratch_local=1gb
#PBS -l walltime=0:30:00
#PBS -m a
# results/*.tar.gz -> results/summary.md and results/site_data.json.gz (parses ~1.5 GB of records).
trap 'clean_scratch' EXIT TERM
WD=/storage/plzen1/home/kasnezde
PROJ=$WD/projects/determinism
PY=$WD/virtualenv/vllm/bin/python   # site_data.py needs `tokenizers`
export HF_HUB_OFFLINE=1
cd "$PROJ" || exit 1
$PY src/analyze.py > /dev/null && $PY src/site_data.py
