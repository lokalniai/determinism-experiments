#!/bin/bash
#PBS -N install_llama
#PBS -l select=1:ncpus=1:ngpus=1:mem=4gb:scratch_local=2gb:gpu_cap="compute_70"
#PBS -l walltime=0:20:00
#PBS -m a
# Installs the CUDA build of llama.app (llama.cpp) into ~/.llama-app and ~/.local/bin/llama.
# Must run on a GPU node: install.sh picks the build by probing the hardware.
# Pin a release with: qsub -v LLAMA_VERSION=bXXXXX install_llama.sh

trap 'clean_scratch' EXIT TERM
test -n "$SCRATCHDIR" || { echo >&2 "no scratch"; exit 1; }
export TMPDIR=$SCRATCHDIR
cd "$SCRATCHDIR"

# On compute nodes $HOME is the node's site-local storage; force plzen1.
export HOME=/storage/plzen1/home/kasnezde
export SKIP_ROCM=1 SKIP_VULKAN=1

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
curl -LsSf https://llama.app/install.sh -o install.sh || exit 2
sh install.sh 2>&1 | tee install.log

# install.sh silently falls back to a CPU build if the CUDA probe fails; refuse to keep that.
if grep -q '^Probing CPU' install.log || ! grep -q '^Probing CUDA' install.log; then
    echo >&2 "CUDA probe failed, not keeping a non-CUDA build"
    rm -rf "$HOME/.llama-app" "$HOME/.local/bin/llama"
    exit 3
fi
"$HOME/.local/bin/llama" version || exit 4
echo "CUDA build installed"
