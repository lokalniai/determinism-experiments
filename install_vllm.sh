#!/bin/bash
#PBS -N install_vllm
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=20gb
#PBS -l walltime=1:30:00
#PBS -m a
# Creates (or upgrades) the `vllm` venv in $WD_VIRTUALENV_DIR with the latest vllm from PyPI.
# Runs on a CPU node: don't use `--torch-backend=auto` here, it would pick CPU torch.

trap 'clean_scratch' EXIT TERM
test -n "$SCRATCHDIR" || { echo >&2 "no scratch"; exit 1; }
export TMPDIR=$SCRATCHDIR

WD=/storage/plzen1/home/kasnezde
export UV_CACHE_DIR=$WD/.cache/uv UV_LINK_MODE=copy
UV=$WD/.local/bin/uv
VENV=$WD/virtualenv/vllm

[ -d "$VENV" ] || $UV venv --python 3.12 "$VENV" || exit 2
VIRTUAL_ENV=$VENV $UV pip install -U vllm pdbpp || exit 3

"$VENV/bin/python" -c 'import torch, vllm; print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)'
