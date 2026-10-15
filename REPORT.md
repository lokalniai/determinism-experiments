# Determinism in LLM inference: vLLM vs llama.cpp on MetaCentrum GPUs

Report, v2, 2026-10-15 (v1 2026-10-12, with more settings; see the plan's status line). Plan:
[EXPERIMENTS.md](EXPERIMENTS.md). Per-setting metrics: [results/summary.md](results/summary.md)
(regenerate with `python src/analyze.py`). Viewer: https://lokalniai.github.io/determinism-experiments/

## TL;DR

1. **At batch size 1, both engines are bit-exact for the dense models (E1).** Greedy and seeded
   sampling alike: 10 repeats × 5 prompts gave one output per prompt. The exception is the NVFP4
   MoE in vLLM: 10 distinct completions out of 10, greedy or sampled, while **llama.cpp runs the
   same weights on the same Blackwell GPU deterministically**.
2. **Sharing the forward pass with other requests breaks determinism (E2).** vLLM batches of 2–8
   matched the batch-1 output in only 2–38 % of runs, with up to 10 distinct completions from 10
   repeats. llama.cpp with 2, 4 or 8 parallel slots shared with concurrent clients: 18–54 %, with 4–10
   distinct completions, and hardly any dependence on the number of slots.
3. **vLLM's batch-invariant mode removes the batching effect, at ~3× the cost (E3).** On Qwen3-4B,
   batches of 1, 2, 4 and 8 all gave the same output, at 49/86/173 tok/s instead of 156/264/512
   (batch 2/4/8). It refuses the other two models (LFM2.5's short-conv layers; the MoE's W4A16 NVFP4
   kernels).
4. **Changing the GPU changes vLLM's output; llama.cpp's did not change (E4).** On L40S instead of
   A100, vLLM diverged on 3 of 5 prompts (LFM2.5) and 4 of 5 (Qwen3-4B), but stayed deterministic
   on the L40S itself. llama.cpp was bit-identical on both GPUs for both dense models.
5. Outputs split where the top-2 tokens are (nearly) tied: the median top-2 logprob gap at the
   first differing token is 0–0.375 in vLLM (multiples of the BF16 step) and 0.006–0.062 in llama.cpp.

## Setup

| | |
|---|---|
| Engines | vLLM 0.31.0 (torch 2.13, CUDA 13.0, FlashInfer sampler off); llama.cpp b11429 (`d812350`), own CUDA 13.1 build for sm_80/86/89/120a |
| Models | `LiquidAI/LFM2.5-350M` (BF16), `Qwen/Qwen3-4B-Instruct-2507` (BF16), `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE; NVFP4 W4A16 experts, FP8 attention projections). GGUFs converted from the same checkpoints. |
| GPUs | A100-SXM4-40GB (zia; reference for the dense models), RTX PRO 6000 Blackwell (grogu; the MoE), L40S (alfrid6; E4). |
| Prompts | 5 targets (`hello` "Hello! Who are you?", `fact`, `story`, `math`, `code`), chat template rendered once to token IDs and fed to both engines; reasoning off; `max_tokens=1024`. 30 filler prompts for E2/E3. |
| Repeats | 10 per (setting, model, prompt). |
| Metrics | **match** = share of runs whose token sequence equals the reference; **outputs** = distinct completions per prompt (max over prompts; 1 = deterministic); **div@** = median first differing token; **gap** = median top-2 logprob gap of the reference at that token. |

Reference: repeat 0 of E1 greedy on the reference GPU, same engine and model. Exceptions: E1
sampling compares with repeat 0 of the same sampling setting, and E3 with repeat 0 of
batch-invariant batch 1.

## Results

Cells are **match / outputs**.

### E1: batch size 1, one process

| Engine | Setting | LFM2.5-350M | Qwen3-4B | Qwen3.6 MoE |
|---|---|---|---|---|
| vLLM | greedy | 1.00 / 1 | 1.00 / 1 | **0.26 / 10** (div@43) |
| vLLM | T = 0.7 | 1.00 / 1 | 1.00 / 1 | **0.22 / 10** |
| vLLM | T = 1.0 | 1.00 / 1 | 1.00 / 1 | **0.10 / 10** |
| llama.cpp | greedy | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 |
| llama.cpp | T = 0.7 | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 |
| llama.cpp | T = 1.0 | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 |

Sampling: top-p 0.95, the same seed in every repeat (vLLM per-request `seed`, llama.cpp `seed`).

### E2: batching

| Engine | Setting | LFM2.5-350M | Qwen3-4B | Qwen3.6 MoE |
|---|---|---|---|---|
| vLLM | batch of 2 | 0.02 / 10 | 0.20 / 8 | 0.10 / 10 |
| vLLM | batch of 4 | 0.06 / 9 | 0.20 / 8 | 0.04 / 10 |
| vLLM | batch of 8 | 0.38 / 7 | 0.20 / 9 | 0.10 / 10 |
| llama.cpp | `--parallel 2` | 0.48 / 6 | 0.42 / 9 | 0.20 / 10 |
| llama.cpp | `--parallel 4` | 0.54 / 5 | 0.46 / 9 | 0.18 / 10 |
| llama.cpp | `--parallel 8` | 0.52 / 4 | 0.44 / 9 | 0.18 / 10 |

vLLM: the target first in an offline batch, the other N−1 prompts drawn at random from the
fillers (a new draw per repeat). llama.cpp: `llama-server --parallel N` with N−1 clients sending
fillers (256 tokens each) concurrently, so the target shares each forward pass with up to N−1
sequences. The `hello` prompt of Qwen3-4B survived every vLLM batch; most other prompts split
within the first 3–90 tokens. In llama.cpp the match rate barely moves between 2, 4 and 8 slots:
what matters is that the target shares a forward pass at all, not with how many sequences.
The `--parallel 2` and `8` runs (2026-10-15) ran on zia1, the reference on zia2; llama.cpp gave
bit-identical output even across A100 and L40S (E4), so the node is not a factor.

### E3: vLLM batch-invariant mode (Qwen3-4B, A100)

| Batch size | default (E1/E2): match / outputs | batch-invariant: match / outputs | default tok/s | batch-invariant tok/s |
|---|---|---|---|---|
| 1 | 1.00 / 1 | 1.00 / 1 | 118 | 39 |
| 2 | 0.20 / 8 | **1.00 / 1** | 156 | 49 |
| 4 | 0.20 / 8 | **1.00 / 1** | 264 | 86 |
| 8 | 0.20 / 9 | **1.00 / 1** | 512 | 173 |

Throughput is the whole batch (at batch size 1, the single request). The batch-invariant output is
fixed but differs from the default batch-1 output on 4 of 5 prompts (only `hello` is equal): it is
a different computation, which is why E3 is compared with its own batch-1 run. Unsupported:
- LFM2.5: `VLLM batch_invariant mode is not supported for SHORT_CONV_ATTN`.
- Qwen3.6 MoE: `CutlassNvFp4LinearKernel does not support W4A16`. Its Gated DeltaNet layers would
  also be rejected: in vLLM 0.31 only the FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION and MLA backends
  declare batch-invariance support.

### E4: L40S instead of A100 (greedy, batch size 1)

| Engine | LFM2.5-350M | Qwen3-4B |
|---|---|---|
| vLLM | 0.40 / 1 | 0.20 / 1 |
| llama.cpp | 1.00 / 1 | 1.00 / 1 |

First differing token, L40S vs A100 (`=` identical over the whole completion):

| | `hello` | `fact` | `story` | `math` | `code` |
|---|---|---|---|---|---|
| vLLM LFM2.5 | = | 72 | 27 | = | 182 |
| vLLM Qwen3-4B | = | 12 | 13 | 263 | 185 |
| llama.cpp, both models | = | = | = | = | = |

Each L40S setting is itself deterministic (1 output per prompt over 10 repeats): the GPU changes
which output you get, not whether it repeats.

## Interpretation

- **Batch-shape dependence is the dominant source of nondeterminism in serving.** Once other
  requests share the forward pass, the reduction order of the GEMM and attention kernels changes
  with the batch shape, and outputs split at the first near-tie. With BF16 logits (vLLM) these are
  often exact ties (gap 0.000) or one BF16 step (0.125).
- **Where the split happens is prompt-dependent, not random.** The short `hello` answer survived
  most perturbations; `story`, `fact` and `math` often split within the first 3–90 tokens.
- **llama.cpp is more stable across hardware** in this setup: its CUDA kernels for sm_80 and sm_89
  gave bit-identical results, while vLLM selects different kernels (and torch.compile/Triton
  autotuning) per GPU.
- **vLLM MoE at batch size 1:** candidate causes, not yet isolated: the FlashInfer GDN prefill
  kernel, the Marlin W4A16 MoE kernel, or expert routing on near-tied router scores. llama.cpp's
  determinism on the same weights and GPU rules out the model itself.

## Limitations

- Small scale: 3 models, 5 prompts, 10 repeats; the match rates are coarse (steps of 0.02–0.2).
- E4 covers one GPU pair (L40S vs A100). A40 and H100 were not scheduled within the 2 h queue
  limit.
- MoE: vLLM and llama.cpp use the same NVFP4 expert weights but different precision for the
  attention projections (vLLM FP8, llama.cpp BF16), and vLLM ignored the checkpoint's FP8
  KV-cache hint (BF16 KV cache).
- E3 exists only for Qwen3-4B, on A100 (vLLM has no tuned batch-invariant matmul configs for sm_80;
  on sm_89/90/120 the throughput cost may differ).
- E2 differs between the engines by design: vLLM batches are offline and fixed per repeat; in
  llama.cpp the fillers arrive concurrently, so the batch composition also varies over time.
- The mid-size model was changed from Qwen3.5-4B to Qwen3-4B-Instruct-2507 to make batch-invariant
  mode possible; Qwen3.5-4B appears only in smoke tests.

## Possible follow-ups

1. Ablate the vLLM MoE batch-size-1 nondeterminism (~30 GPU-min on grogu): `enforce_eager`, Triton
   GDN prefill, another NVFP4 MoE backend.
2. Batch-invariant mode on L40S/Blackwell, where vLLM has tuned configs, to measure its cost there.
3. More GPU types for E4, and a longer prompt set for tighter estimates.

## Reproducing

```bash
qsub jobs/prepare.sh            # download models, convert GGUFs (models/manifest.tsv)
qsub jobs/build_llama.sh        # llama.cpp b11429 for sm_80/86/89/120a
qsub jobs/pack_venv.sh          # vLLM venv tarball for fast job start
python src/render_prompts.py    # prompts -> token IDs (prompts/rendered/)
qsub -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=40gb:cluster=zia \
     -v ENGINE=llama+vllm,MODELS=lfm350m+q3_4b,EXPS=e1+e1_t07+e1_t10 jobs/run.sh
qsub jobs/analyze.sh            # results/*.tar.gz -> results/summary.md, results/site_data.json.gz
python src/build_site.py --out site
```

Setting names are in the docstrings of `src/run_vllm.py` and `src/run_llama.py`: `e1*` exist in
both, `e2_bs*` and `e3_*` only in vLLM (`ENGINE=vllm`), `e2_np*` only in llama.cpp (`ENGINE=llama`).
E4 is `e1` submitted with `cluster=alfrid`.

Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the
Ministry of Education, Youth and Sports of the Czech Republic.
