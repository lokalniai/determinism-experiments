# Determinism in LLM inference: vLLM vs llama.cpp on MetaCentrum GPUs

Report, 2026-10-12. Plan: [EXPERIMENTS.md](EXPERIMENTS.md). Raw per-run tables:
[results/summary.md](results/summary.md) (regenerate with `python src/analyze.py`).

## TL;DR

1. **At batch size 1, both engines are bit-exact for the dense models.** Every greedy and seeded run
   (10 repeats × 5 prompts) was identical within a process, across restarts and across nodes of the
   same GPU type. The one exception is the MoE in vLLM (see 3).
2. **Batching with other requests is what breaks determinism.** With fillers in the same batch, vLLM
   produced up to 10 different completions from 10 repeats; llama.cpp with `--parallel 4` and
   concurrent clients, 5–10. With one slot (`--parallel 1`) llama.cpp stays exact even under load,
   because requests are processed one at a time.
3. **The NVFP4 MoE (Qwen3.6-35B-A3B) is nondeterministic in vLLM even at batch size 1**: 10 distinct
   completions out of 10, already within the same process, while **llama.cpp runs the same weights
   on the same Blackwell GPU deterministically**. This is specific to vLLM's kernels for this model.
4. **vLLM's batch-invariant mode works, but costs ~3× throughput and supports few architectures.**
   On Qwen3-4B it made every batch size (1–8) and batch position give the same output, at
   173 vs 512 tok/s (batch 8) and 39 vs 118 tok/s (batch 1). It refuses the hybrid LFM2.5
   (short-conv layers) and Qwen3.6 (Gated DeltaNet layers, and W4A16 NVFP4).
5. **Changing the GPU changes vLLM's output; llama.cpp's output did not change.** vLLM on L40S vs
   A100 diverged on 3 of 5 prompts (LFM2.5) and 4 of 5 (Qwen3-4B). llama.cpp was bit-identical on both GPUs for both
   dense models.
6. **Engine settings shift the output but keep it stable.** CUDA graphs vs eager, attention backend,
   flash attention on/off and prefix caching give a *different but reproducible* text (1 distinct
   output over 10 repeats in all but one case).
7. Divergence starts where the top-2 tokens are (nearly) tied: the median top-2 logprob gap at the
   first differing token is 0–0.375 in vLLM (multiples of the BF16 step) and 0.002–0.06 in llama.cpp.

## Setup

| | |
|---|---|
| Engines | vLLM 0.31.0 (torch 2.13, CUDA 13.0, FlashInfer sampler off); llama.cpp b11429 (`d812350`), own CUDA 13.1 build for sm_80/86/89/120a |
| Models | `LiquidAI/LFM2.5-350M` (BF16), `Qwen/Qwen3-4B-Instruct-2507` (BF16), `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE; NVFP4 W4A16 experts, FP8 attention projections). GGUFs converted from the same checkpoints. |
| GPUs | A100-SXM4-40GB (zia, reference), L40S (alfrid6), RTX PRO 6000 Blackwell (grogu; MoE only). A40 (galdor) was skipped: not scheduled within the 2 h queue limit. |
| Prompts | 5 targets (`hello` "Hello! Who are you?", `fact`, `story`, `math`, `code`), chat template rendered once to token IDs and fed to both engines; reasoning off; `max_tokens=1024`. 30 filler prompts for batching/load. |
| Repeats | 10 per (model, experiment, config). |
| Metrics | **match** = share of runs whose token sequence equals the reference run; **uniq** = distinct completions per prompt (max over prompts; 1 = deterministic); **div@** = median first differing token; **gap** = median top-2 logprob gap of the reference at that token. |
| Cost | 10.3 GPU-h + 2.1 CPU-job-h (incl. smoke tests, build, conversion). |

Reference for each (GPU, engine, model) is repeat 0 of E1a (batch size 1, one process). For E5 it is
repeat 0 of the same sampling setting, and for batch-invariant runs repeat 0 at batch size 1.

## Results

### RQ1: repeatability at batch size 1 (E1)

| GPU | Engine | Model | within process (E1a) | 10 restarts (E1b) | other node (E1c) |
|---|---|---|---|---|---|
| A100 | vLLM | LFM2.5-350M | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 (zia2 vs zia4) |
| A100 | vLLM | Qwen3-4B | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 (same node) |
| A100 | llama.cpp | LFM2.5-350M | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 (same node) |
| A100 | llama.cpp | Qwen3-4B | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 (same node) |
| L40S | both | both dense | 1.00 / 1 | | |
| Blackwell | llama.cpp | Qwen3.6 MoE | 1.00 / 1 | 1.00 / 1 | |
| Blackwell | **vLLM** | **Qwen3.6 MoE** | **0.26 / 10** (div@43) | **0.18 / 10** | |

Cells: match / uniq. E1c landed on the reference node zia2 for three of the four combinations;
a run pinned to zia3 did not start within the 2 h queue limit (see Limitations).

### RQ2: batching and concurrency (E2)

**vLLM, offline batches** (target + random fillers; match / uniq):

| Batch size | LFM2.5 (target first / last) | Qwen3-4B (first / last) | Qwen3.6 MoE (first / last) |
|---|---|---|---|
| 2 | 0.02 / 10 · 0.24 / 8 | 0.20 / 8 · 0.20 / 7 | 0.10 / 10 · 0.08 / 10 |
| 4 | 0.06 / 9 · 0.40 / 10 | 0.20 / 8 · 0.20 / 9 | 0.04 / 10 · 0.00 / 10 |
| 8 | 0.38 / 7 · 0.32 / 8 | 0.20 / 9 · 0.20 / 10 | 0.10 / 10 · 0.08 / 10 |

**Server load** (`vllm serve`, `llama-server`; match / uniq):

| | LFM2.5 | Qwen3-4B | Qwen3.6 MoE |
|---|---|---|---|
| vLLM, no concurrent clients | 1.00 / 1 | 1.00 / 1 | 0.20 / 10 |
| vLLM, 4 filler clients | 0.38 / 10 | 0.20 / 10 | 0.06 / 10 |
| llama.cpp `--parallel 1`, 3 filler clients | 1.00 / 1 | 1.00 / 1 | 1.00 / 1 |
| llama.cpp `--parallel 4`, 3 filler clients | 0.54 / 5 | 0.46 / 9 | 0.18 / 10 |

**Prefix caching** (vLLM, cache on; repeat 0 cold, 1–9 warm): LFM2.5 0.40 / 1 (cache on changes the
output, but consistently); Qwen3-4B 0.28 / 2 (cold and warm differ from each other).

**Batch-invariant mode** (`VLLM_BATCH_INVARIANT=1`, Qwen3-4B, A100):

| Batch size | default: match / uniq | batch-invariant: match / uniq | default tok/s | batch-invariant tok/s |
|---|---|---|---|---|
| 1 | 1.00 / 1 | 1.00 / 1 | 118 | 39 |
| 2 | 0.20 / 8 | **1.00 / 1** | 156 | 49 |
| 4 | 0.20 / 9 | **1.00 / 1** | 264 | 86 |
| 8 | 0.20 / 10 | **1.00 / 1** | 512 | 173 |

(Throughput at batch size > 1 is the whole batch; at batch size 1 a single request.) The
batch-invariant output differs from the default output on 4 of 5 prompts (only `hello` is equal):
it is a different, but fixed, computation. Unsupported models:
- LFM2.5: `VLLM batch_invariant mode is not supported for SHORT_CONV_ATTN`.
- Qwen3.6 MoE: `CutlassNvFp4LinearKernel does not support W4A16`. Its Gated DeltaNet layers would
  also be rejected: in vLLM 0.31 only the FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION and MLA backends
  declare batch-invariance support.

### RQ3: engine settings (E3, batch size 1, A100)

| Setting | LFM2.5 | Qwen3-4B |
|---|---|---|
| vLLM `enforce_eager` (no CUDA graphs / torch.compile) | 0.00 / 1 | 0.20 / 1 |
| vLLM `TRITON_ATTN` instead of `FLASH_ATTN` | 0.20 / 1 | 0.20 / 1 |
| llama.cpp `-fa off` | 0.80 / 1 | 0.80 / 1 |

Each setting gives a different output from the reference, but the same output in all 10 repeats.

### RQ4: hardware and engine (E4; first differing token, E1a repeat 0)

| | `hello` | `fact` | `story` | `math` | `code` |
|---|---|---|---|---|---|
| vLLM LFM2.5, L40S vs A100 | = | 72 | 27 | = | 182 |
| vLLM Qwen3-4B, L40S vs A100 | = | 12 | 13 | 263 | 185 |
| llama.cpp LFM2.5, L40S vs A100 | = | = | = | = | = |
| llama.cpp Qwen3-4B, L40S vs A100 | = | = | = | = | = |
| A100: llama.cpp vs vLLM, LFM2.5 | 46 | 30 | 172 | = | 50 |
| A100: llama.cpp vs vLLM, Qwen3-4B | = | 89 | 13 | 3 | 240 |
| Blackwell: llama.cpp vs vLLM, MoE | = | 43 | 37 | 239 | 38 |

(`=` identical over the whole completion.) The MoE engine comparison is not like-for-like: vLLM
runs the attention projections in FP8, llama.cpp in BF16 (the converter dequantized them).

### RQ5: seeded sampling (E5, batch size 1; T = 0.7 / 1.0, top-p 0.95, fixed seed)

Reproducible (1.00 / 1) for every dense model in both engines and for the MoE in llama.cpp.
vLLM MoE: 0.22 / 10 and 0.10 / 10, consistent with its greedy nondeterminism.

## Interpretation

- **Batch-size dependence is the dominant source of nondeterminism in serving.** Once other requests
  share the forward pass, the GEMM/attention reduction order changes with batch shape, and outputs
  split at the first near-tie. With BF16 logits (vLLM) these are often exact ties (gap 0.000) or
  one BF16 step (0.125).
- **Where the split happens is prompt-dependent, not random.** The short `hello` answer survived almost
  every perturbation; `story`, `fact` and `math` split within the first 3–90 tokens in many cases.
- **llama.cpp is more stable across hardware** in this setup: its CUDA kernels for sm_80 and sm_89
  produced bit-identical results, while vLLM selects different kernels (and torch.compile/Triton
  autotuning) per GPU.
- **vLLM MoE at batch size 1:** candidate causes, not yet isolated: the FlashInfer GDN prefill
  kernel, the Marlin W4A16 MoE kernel, or expert routing on near-tied router scores. llama.cpp's
  determinism on the same weights and GPU rules out the model itself.

## Limitations

- Small scale: 3 models, 5 prompts, 10 repeats, 2–3 GPU types; the match rates are coarse
  (steps of 0.02–0.2).
- A40 (sm_86) and H100 (sm_90) are missing (queue limit / availability).
- E1c covers a different node only for vLLM LFM2.5 (zia4 vs zia2); the other three ran on the same
  node as their reference. A run pinned to zia3 (job 24608340) was cancelled after 2 h in the queue.
- MoE: the vLLM and llama.cpp runs use the same NVFP4 expert weights but different precision for
  attention projections, and vLLM ignored the checkpoint's FP8 KV-cache hint (BF16 KV cache).
- Batch-invariant results exist only for Qwen3-4B, on A100 (vLLM has no tuned batch-invariant matmul
  configs for sm_80; on sm_89/90/120 the throughput cost may differ).
- The mid-size model was changed from Qwen3.5-4B to Qwen3-4B-Instruct-2507 to make batch-invariant
  mode possible; Qwen3.5-4B appears only in smoke tests.

## Possible follow-ups

1. Ablate the vLLM MoE batch-size-1 nondeterminism (~30 GPU-min on grogu): `enforce_eager`, Triton
   GDN prefill, another NVFP4 MoE backend.
2. Batch-invariant mode on L40S/Blackwell, where vLLM has tuned configs, to measure its cost there.
3. Add A40/H100 when the queues allow, and a longer prompt set for tighter estimates.

## Reproducing

```bash
qsub jobs/prepare.sh            # download models, convert GGUFs (models/manifest.tsv)
qsub jobs/build_llama.sh        # llama.cpp b11429 for sm_80/86/89/120a
qsub jobs/pack_venv.sh          # vLLM venv tarball for fast job start
python src/render_prompts.py    # prompts -> token IDs (prompts/rendered/)
qsub -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=40gb:cluster=zia \
     -v ENGINE=llama+vllm,MODELS=lfm350m+q3_4b,EXPS=e1a+e1b jobs/run.sh
python src/analyze.py           # results/*.tar.gz -> results/summary.md
```

Experiment names and settings are in the docstrings of `src/run_vllm.py` and `src/run_llama.py`.
