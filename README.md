# Determinism in LLM inference: vLLM vs llama.cpp

Repeated runs, scores and an interactive viewer for experiments on whether the
same input gives the same output in LLM inference engines, run on MetaCentrum
GPUs.

- [💻 **Interactive viewer of the runs**](https://lokalniai.github.io/determinism-experiments/): one pie per (setting,
  model, prompt) showing how many of the 10 runs match the reference, and the
  throughput cost of vLLM's batch-invariant mode; click a pie for the outputs
- [🖋 **Blogpost (in Czech)**](https://lokalni.ai/blog/determinismus/)
- [📄 **Report**](REPORT.md) · [experiment plan](EXPERIMENTS.md)

---

## What was run

Five fixed prompts (`hello`, `fact`, `story`, `math`, `code`). The chat template
was rendered once to token IDs, and both engines received exactly those IDs.
Reasoning off, `max_tokens=1024`, greedy unless stated, **10 repeats** per
setting.

| model | id | engines |
|---|---|---|
| LFM2.5-350M | `LiquidAI/LFM2.5-350M` | vLLM (BF16), llama.cpp (BF16 GGUF from the same checkpoint) |
| Qwen3-4B | `Qwen/Qwen3-4B-Instruct-2507` | vLLM (BF16), llama.cpp (BF16 GGUF from the same checkpoint) |
| Qwen3.6-35B-A3B | `nvidia/Qwen3.6-35B-A3B-NVFP4` | vLLM, llama.cpp (NVFP4 GGUF from the same checkpoint) |

| engine | version |
|---|---|
| vLLM | 0.31.0 (torch 2.13, CUDA 13.0) |
| llama.cpp | b11429 (`d812350`), CUDA 13.1 build for sm_80/86/89/120a |

GPUs: A100-SXM4-40GB (reference for the dense models), RTX PRO 6000 Blackwell (the MoE), and
L40S (E4).

| experiment | what changes | reference |
|---|---|---|
| E1 | nothing: batch size 1, 10 repeats in one process; greedy and seeded sampling (T = 0.7 / 1.0) | repeat 0 |
| E2 | batching: vLLM offline batches of 2/4/8, llama.cpp `--parallel 2/4/8` with concurrent fillers | E1 greedy |
| E3 | vLLM batch-invariant mode (`VLLM_BATCH_INVARIANT=1`) at batch 1/2/4/8, Qwen3-4B only | its batch 1 |
| E4 | the GPU: L40S instead of A100, dense models | E1 greedy on A100 |

## Data

- `results/summary.md`: every metric, as produced by `src/analyze.py` over the
  raw runs.
- `results/records.jsonl.gz`: every completion of E1–E4 (2,200 of them):
  token IDs, text, and the setting it came from. The top-20 logprobs at every
  token are left out (1.5 GB).
- `results/site_data.json.gz`: the reduction the viewer is built from
  (`src/site_data.py`): per setting and prompt, the distinct completions, which
  repeats produced them and where each departs from the reference. Its metrics
  come from the same code as `results/summary.md`.

An earlier version of the plan had more settings (restarts, prefix caching,
server load, engine knobs). Those runs are not part of the analysis or of this
repository. Rebuild the viewer with `python src/build_site.py --out <dir>`
(standard library only). The scripts that ran the experiments are in `src/` and
`jobs/`; the [report](REPORT.md#reproducing) lists the steps.

Computational resources were provided by the e-INFRA CZ project (ID:90254),
supported by the Ministry of Education, Youth and Sports of the Czech Republic.
