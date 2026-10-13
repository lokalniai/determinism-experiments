# Determinism in LLM inference: vLLM vs llama.cpp

Repeated runs, scores and an interactive viewer for experiments on whether the
same input gives the same output in LLM inference engines, run on MetaCentrum
GPUs.

- [💻 **Interactive viewer of the runs**](https://lokalniai.github.io/determinism-experiments/)
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

GPUs: A100-SXM4-40GB (reference), L40S, and RTX PRO 6000 Blackwell (MoE only).

| experiment | what changes |
|---|---|
| E1 | nothing: repeats in one process, across restarts, as a separate job |
| E2 | batching with filler prompts, concurrent server load, prefix caching, vLLM batch-invariant mode |
| E3 | engine settings: CUDA graphs, attention backend, flash attention |
| E4 | GPU type, and engine vs engine |
| E5 | seeded sampling (T = 0.7 / 1.0) |

## Data

- `results/summary.md`: every metric, as produced by `src/analyze.py` over the
  raw runs.
- `results/records.jsonl.gz`: every completion (4,100 of them): token IDs,
  text, and the setting it came from. The top-20 logprobs at every token are
  left out (1.5 GB).
- `results/site_data.json.gz`: the reduction the viewer is built from
  (`src/site_data.py`). It holds the distinct completions, where each one
  departs from its reference, and the top candidates at that point. Its metrics
  are checked against `results/summary.md` when it is built.

Rebuild the viewer with `python src/build_site.py --out <dir>` (standard library
only). The scripts that ran the experiments are in `src/` and `jobs/`; the
[report](REPORT.md#reproducing) lists the steps.

Computational resources were provided by the e-INFRA CZ project (ID:90254),
supported by the Ministry of Education, Youth and Sports of the Czech Republic.
