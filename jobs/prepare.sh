#!/bin/bash
#PBS -N det_prepare
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=60gb
#PBS -l walltime=3:00:00
#PBS -m a
# Downloads the three models into $HF_HOME and converts each one to GGUF with the llama.cpp
# converter from the same commit as the installed binary (b11429). GGUFs go to models/,
# with models/manifest.tsv (HF revision, file, sha256).
# Submit from the project root: qsub -o logs/ -e logs/ jobs/prepare.sh

trap 'clean_scratch' EXIT TERM
test -n "$SCRATCHDIR" || { echo >&2 "no scratch"; exit 1; }
export TMPDIR=$SCRATCHDIR

WD=/storage/plzen1/home/kasnezde
PROJ=$WD/projects/determinism
export HF_HOME=$WD/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1
# The system profile sets HF_HUB_CACHE to a scratch dir, which wins over HF_HOME; pin it.
export HF_HUB_CACHE=$HF_HOME/hub
PY=$WD/virtualenv/vllm/bin/python
HF=$WD/virtualenv/vllm/bin/hf
LLAMA_TAG=b11429
mkdir -p "$PROJ/models"
cd "$SCRATCHDIR"

echo "host: $(hostname)  start: $(date -Is)"
df -h "$WD" | tail -1

git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp || exit 2
git -C llama.cpp rev-parse HEAD

# repo  gguf-name  [fallback-repo  fallback-file]
convert() {
    local repo=$1 name=$2 snap rev out=$SCRATCHDIR/$2.gguf
    snap=$($HF download "$repo" --quiet) || { echo >&2 "download failed: $repo"; return 1; }
    rev=$(basename "$snap")
    echo "== $repo @ $rev -> $name.gguf ($(du -sh -L "$snap" | cut -f1)) in $snap"
    [[ $(realpath "$snap") == "$(realpath "$HF_HUB_CACHE")"/* ]] ||{ echo >&2 "snapshot outside $HF_HUB_CACHE"; return 1; }
    if [ -f "$PROJ/models/$name.gguf" ]; then echo "   $name.gguf exists, skipping conversion"; return 0; fi
    # bf16 keeps unquantized tensors bit-identical to the checkpoint; NVFP4 tensors are repacked as-is
    if $PY llama.cpp/convert_hf_to_gguf.py "$snap" --outtype bf16 --outfile "$out" > "$name.convert.log" 2>&1; then
        cp "$out" "$PROJ/models/" || return 1
        printf '%s\t%s\t%s\t%s\n' "$repo" "$rev" "$name.gguf" "$(sha256sum "$out" | cut -d' ' -f1)" >> "$PROJ/models/manifest.tsv"
        rm -f "$out"
    else
        echo >&2 "conversion failed for $repo, tail of log:"; tail -20 "$name.convert.log" >&2
        return 1
    fi
    cp "$name.convert.log" "$PROJ/models/"
}

status=0
convert LiquidAI/LFM2.5-350M lfm2.5-350m-bf16 || status=3
convert Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct-2507-bf16 || status=3
if ! convert nvidia/Qwen3.6-35B-A3B-NVFP4 qwen3.6-35b-a3b-nvfp4; then
    # Fallback (see EXPERIMENTS.md §3): separately quantized NVFP4 GGUF, different weights than vLLM
    echo "== NVFP4 conversion failed, downloading fallback GGUF"
    f=$($HF download michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf --quiet) || status=4
    [ -n "$f" ] && printf '%s\t%s\t%s\t%s\n' michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF fallback "$f" "$(sha256sum "$f" | cut -d' ' -f1)" >> "$PROJ/models/manifest.tsv"
fi

ls -la "$PROJ/models"
df -h "$WD" | tail -1
echo "end: $(date -Is)  status: $status"
exit $status
