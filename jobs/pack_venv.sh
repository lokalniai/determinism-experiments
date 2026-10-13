#!/bin/bash
#PBS -N det_pack_venv
#PBS -l select=1:ncpus=2:mem=8gb:scratch_local=40gb
#PBS -l walltime=1:30:00
#PBS -m a
# Packs the `vllm` venv and its uv-managed base Python (which lives on brno11-elixir) into one tar,
# env/vllm-env.tar. Jobs unpack it to local scratch: importing vLLM from NFS took ~15 min per process.
# Repack after changing the venv.

trap 'clean_scratch' EXIT TERM
test -n "$SCRATCHDIR" || { echo >&2 "no scratch"; exit 1; }
PROJ=/storage/plzen1/home/kasnezde/projects/determinism
PYBASE=/storage/brno11-elixir/home/kasnezde/.local/share/uv/python
VENVS=/storage/plzen1/home/kasnezde/virtualenv
mkdir -p "$PROJ/env"
cd "$SCRATCHDIR"
date -Is
tar cf vllm-env.tar -C "$PYBASE" cpython-3.12.15-linux-x86_64-gnu -C "$VENVS" vllm || exit 2
ls -la vllm-env.tar
date -Is
cp vllm-env.tar "$PROJ/env/vllm-env.tar.part" && mv "$PROJ/env/vllm-env.tar.part" "$PROJ/env/vllm-env.tar" || exit 3
date -Is
