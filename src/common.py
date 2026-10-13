"""Shared pieces for both engine runners: models, prompts, environment fingerprint, output."""
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
HF_HUB_CACHE = Path(os.environ.get("DET_HF_HUB_CACHE", "/storage/plzen1/home/kasnezde/.cache/huggingface/hub"))

# key -> HF repo, pinned revision (see models/manifest.tsv), GGUF file, vLLM extra kwargs
MODELS = {
    "lfm350m": dict(
        repo="LiquidAI/LFM2.5-350M",
        rev="9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
        gguf="lfm2.5-350m-bf16.gguf",
        vllm={},
    ),
    # Standard attention, so vLLM's batch-invariant mode works (it rejects the hybrid
    # Qwen3.5 GDN and LFM2 short-conv layers). Replaced Qwen/Qwen3.5-4B ("qwen4b") on 2026-10-12.
    "q3_4b": dict(
        repo="Qwen/Qwen3-4B-Instruct-2507",
        rev="cdbee75f17c01a7cc42f958dc650907174af0554",
        gguf="qwen3-4b-instruct-2507-bf16.gguf",
        vllm={},
    ),
    "qwen35moe": dict(
        repo="nvidia/Qwen3.6-35B-A3B-NVFP4",
        rev="1355db6a052410cfd62085d94b58866fd0f2c3c5",
        gguf="qwen3.6-35b-a3b-nvfp4.gguf",
        vllm=dict(limit_mm_per_prompt={"image": 0, "video": 0}),
    ),
}

MAX_TOKENS = 1024
TOP_LOGPROBS = 20
SEED = 1234


def hf_path(model: str) -> Path:
    m = MODELS[model]
    return HF_HUB_CACHE / ("models--" + m["repo"].replace("/", "--")) / "snapshots" / m["rev"]


def gguf_path(model: str) -> Path:
    return Path(os.environ.get("DET_MODELS_DIR", PROJ / "models")) / MODELS[model]["gguf"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rendered_prompts(model: str, kind: str = "targets"):
    """Chat-templated token IDs produced by src/render_prompts.py; identical input for both engines."""
    return load_jsonl(PROJ / "prompts" / "rendered" / f"{model}.{kind}.jsonl")


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001 - fingerprint is best effort
        return f"ERROR: {e}"


def env_info(**extra):
    gpu = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total,uuid,pci.bus_id",
                "--format=csv,noheader"])
    smi = _cmd(["nvidia-smi"])
    cuda_driver = next((part.split(":")[1].strip().split()[0] for part in smi.split("|")
                        if "CUDA Version" in part), None)
    return dict(
        host=socket.gethostname(),
        pbs_jobid=os.environ.get("PBS_JOBID"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=gpu,
        cuda_driver_version=cuda_driver,
        python=platform.python_version(),
        time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    )


class Writer:
    """Appends one JSON record per (prompt, repeat) to <outdir>/<exp>.<engine>.<model>.jsonl."""

    def __init__(self, outdir, exp, engine, model, env):
        self.path = Path(outdir) / f"{exp}.{engine}.{model}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base = dict(exp=exp, engine=engine, model=model)
        self.f = open(self.path, "a")
        (Path(outdir) / f"env.{exp}.{engine}.{model}.json").write_text(json.dumps(env, indent=1))

    def write(self, **rec):
        self.f.write(json.dumps({**self.base, **rec}) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
