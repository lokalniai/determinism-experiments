#!/bin/bash
#PBS -N det_build_llama
#PBS -l select=1:ncpus=8:mem=24gb:scratch_local=20gb
#PBS -l walltime=2:00:00
#PBS -m a
# Builds llama-server from llama.cpp b11429 (same commit as the llama.app binary) for
# sm_80, sm_86, sm_89 and sm_120a inside the NGC PyTorch 26.02 container (CUDA 13.1).
# The llama.app binary only contains sm_80 code and fails on Blackwell (grogu).
# Result: llama.cpp-b11429/bin/llama-server; run it through bin/llama (wrapper, same container).

trap 'clean_scratch' EXIT TERM
test -n "$SCRATCHDIR" || { echo >&2 "no scratch"; exit 1; }
export TMPDIR=$SCRATCHDIR

PROJ=/storage/plzen1/home/kasnezde/projects/determinism
IMAGE=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:26.02-py3.SIF
TAG=b11429
cd "$SCRATCHDIR"
git clone -q --depth 1 --branch $TAG https://github.com/ggml-org/llama.cpp || exit 2

singularity exec -B "$SCRATCHDIR" "$IMAGE" bash -c '
set -e
cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="80-real;86-real;89-real;120a-real" \
      -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j "$NCPUS"
' > build.log 2>&1 || { tail -40 build.log >&2; exit 3; }

OUT=$PROJ/llama.cpp-$TAG
mkdir -p "$OUT/bin"
cp llama.cpp/build/bin/llama-server "$OUT/bin/" || exit 4
cp build.log "$OUT/"
git -C llama.cpp rev-parse HEAD > "$OUT/COMMIT"
echo "$IMAGE" > "$OUT/IMAGE"
singularity exec "$IMAGE" bash -c 'cuobjdump --list-elf '"$OUT"'/bin/llama-server | sed -E "s/.*\.(sm_[0-9a-z]+)\..*/\1/" | sort | uniq -c' | tee "$OUT/ARCHS"
ls -la "$OUT/bin"
