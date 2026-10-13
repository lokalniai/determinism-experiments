#!/bin/bash
#PBS -N det_run
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=40gb
#PBS -l walltime=1:00:00
#PBS -m a
# Runs experiments for one engine on one GPU. Pass via -v:
#   ENGINE=llama|vllm|llama+vllm  MODELS=lfm350m+q3_4b  EXPS=e1a+e1b+...
# GPU type is chosen at submit time, e.g. -l select=...:cluster=fobos (see jobs/submit.sh).
# Results: results/<jobnum>.<engine>.<cluster>.tar.gz

trap 'clean_scratch' EXIT TERM
test -n "$SCRATCHDIR" || { echo >&2 "no scratch"; exit 1; }
export TMPDIR=$SCRATCHDIR

WD=/storage/plzen1/home/kasnezde
PROJ=$WD/projects/determinism
export HF_HOME=$WD/.cache/huggingface HF_HUB_CACHE=$WD/.cache/huggingface/hub HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$SCRATCHDIR/cache VLLM_CACHE_ROOT=$SCRATCHDIR/vllm-cache TRITON_CACHE_DIR=$SCRATCHDIR/triton
# PyTorch sampler instead of FlashInfer's, which JIT-compiles a kernel in every job
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=$WD/virtualenv/vllm/bin/python

: "${ENGINE:?}" "${MODELS:?}" "${EXPS:?}"
MODELS=${MODELS//+/,} EXPS=${EXPS//+/,}   # qsub -v splits on commas, so lists use '+'

if [[ $ENGINE == *vllm* ]]; then
    # Importing vLLM from NFS takes ~15 min per process; unpack venv + base Python to local scratch.
    tar xf "$PROJ/env/vllm-env.tar" -C "$SCRATCHDIR" || { echo >&2 "venv unpack failed"; exit 2; }
    V=$SCRATCHDIR/vllm PYBASE=$SCRATCHDIR/cpython-3.12.15-linux-x86_64-gnu
    ln -sfn "$PYBASE/bin/python3.12" "$V/bin/python"
    sed -i "s|^home = .*|home = $PYBASE/bin|" "$V/pyvenv.cfg"
    export VIRTUAL_ENV=$V PATH=$V/bin:$PATH   # ninja/nvcc for FlashInfer JIT
    PY=$V/bin/python
    echo "venv unpacked: $(date -Is)"
fi
JOBNUM=${PBS_JOBID%%.*}
CLUSTER=$(hostname -s | sed 's/[0-9-]*$//')
NAME=$JOBNUM.${ENGINE//+/-}.$CLUSTER
OUT=$SCRATCHDIR/$NAME
mkdir -p "$OUT"
cd "$SCRATCHDIR"

echo "host: $(hostname)  engine: $ENGINE  models: $MODELS  exps: $EXPS  start: $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader
nvidia-smi > "$OUT/nvidia-smi.txt"

rc=0
for eng in ${ENGINE//+/ }; do
    $PY "$PROJ/src/run_$eng.py" --models "$MODELS" --exps "$EXPS" --out "$OUT" >> "$OUT/run.log" 2>&1 || rc=$?
done
grep -E 'FAILED|Traceback|Error' "$OUT/run.log" | head -20
tail -3 "$OUT/run.log"

mkdir -p "$PROJ/results"
tar czf "$PROJ/results/$NAME.tar.gz" -C "$SCRATCHDIR" "$NAME" || { trap - EXIT; echo >&2 "copy-back failed, scratch kept on $(hostname)"; exit 3; }
echo "end: $(date -Is)  rc: $rc  -> results/$NAME.tar.gz"
exit $rc
