# Determinism in LLM inference: experiment plan

Status: **v3, 2026-10-12; experiments completed, results in [REPORT.md](REPORT.md).** Comment inline (`> ZK: ...`) or edit freely.

Installed: vLLM 0.31.0 (torch 2.13.0+cu130, venv `vllm`), llama.app b11429 (`~/.llama-app/llama`,
installed on zia4/A100). Scope: small scale, both characterization and "what does determinism
cost". Output: a report in Markdown. Single GPU only (no tensor parallelism).

## 1. Research questions

- **RQ1 (repeatability):** Is greedy decoding at batch size 1 bit-identical across repeated runs
  with the same engine, config and GPU? (vLLM and llama.cpp)
- **RQ2 (batch invariance):** How much does the output for a fixed prompt depend on what else is
  being processed (batch size, concurrent requests, prefix cache)? Does vLLM's batch-invariant mode
  remove the effect, and what does it cost in throughput? (mainly vLLM, plus a small llama.cpp
  `--parallel` check)
- **RQ3 (engine knobs):** Do CUDA graphs, the attention backend (vLLM) or flash attention
  (llama.cpp) change the output?
- **RQ4 (hardware):** How far do outputs drift across GPU types, and across nodes with the same GPU?
- **RQ5 (sampling):** Is seeded sampling (T > 0) reproducible at batch size 1?

Sampling and batching are tested separately (RQ2 is greedy only, RQ5 is batch size 1 only).

## 2. What we measure

Every (prompt, run) record stores the generated token IDs and the top-20 logprobs at each step.
Compared against a reference run:

| Metric | Meaning |
|---|---|
| exact-match rate | fraction of runs whose token sequence equals the reference |
| first divergence index | token position of the first mismatch (∞ if identical); report the distribution |
| # unique completions | per prompt across the 10 repeats (1 = deterministic) |

The stored logprobs are used for a supporting analysis: the gap between the top-2 tokens at
the divergence point (whether outputs split only on near-ties).

## 3. Models

All are ungated on Hugging Face, so no token is needed. Both engines get weights converted from the
**same checkpoint**, so engine differences come from kernels, not from different quantizations.

| Role | vLLM | llama.cpp | Size | GPUs |
|---|---|---|---|---|
| tiny dense | `LiquidAI/LFM2.5-350M` (BF16) | BF16 GGUF, converted by us | ~0.7 GB | all E4 GPUs |
| mid dense | `Qwen/Qwen3-4B-Instruct-2507` (BF16) | BF16 GGUF, converted by us | ~8 GB | all E4 GPUs |
| MoE | `nvidia/Qwen3.6-35B-A3B-NVFP4` | NVFP4 GGUF, converted by us from the same checkpoint | ~23 GB | **grogu only** (Blackwell, native FP4) |

- Conversion uses `convert_hf_to_gguf.py` from the llama.cpp source at the commit that matches the
  binary (b11429). If the NVFP4 (ModelOpt) checkpoint doesn't convert, the fallback is
  `michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF` (HQ). That file is quantized separately from the
  BF16 base with its own imatrix, so the two engines would then run different weights; I'd note
  that in the report.
- MoE precision (checked 2026-10-12): the NVIDIA checkpoint is mixed precision. Routed and shared
  experts and `lm_head` are W4A16 NVFP4 (weight-only), and the GGUF repacks exactly these 241
  tensors. Attention and linear-attention projections are FP8 with an FP8 KV cache in the
  checkpoint; the converter dequantized them to BF16. vLLM runs them in FP8, llama.cpp in BF16, so
  the **MoE cross-engine comparison is not like-for-like**. Observed in vLLM E0 on grogu:
  `quantization=modelopt_mixed`, NVFP4 experts via the Marlin W4A16 kernel (not native FP4),
  FP8 projections via CUTLASS FP8 GEMM, `kv_cache_dtype=auto` (BF16; the FP8 KV hint is not applied).
- Architecture note: Qwen3.6 mixes full attention with Gated DeltaNet linear attention and is a
  vision-language model (vLLM loads it with `language_model_only`, and the GGUF conversion skips
  the vision tower). LFM2.5 mixes convolution and attention layers. Qwen3-4B-Instruct-2507 is a
  standard transformer.
- **Batch-invariant mode needs standard attention.** vLLM 0.31 supports `VLLM_BATCH_INVARIANT=1`
  only for the FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION and MLA backends. It refuses LFM2's
  short-conv layers (`SHORT_CONV_ATTN`, seen on A100) and Qwen3.5/3.6's Gated DeltaNet layers.
  That is why the mid-size model was changed on 2026-10-12 from `Qwen/Qwen3.5-4B` (hybrid, GDN) to
  `Qwen/Qwen3-4B-Instruct-2507`. Batch-invariant results therefore exist only for this model; for
  the other two the report records the limitation. The MoE fails even earlier (grogu, 2026-10-12):
  batch-invariant mode selects `CutlassNvFp4LinearKernel`, which raises "does not support W4A16".
- Disk: about 0.7×2 + 8×2 + 23×2 ≈ 65 GB (plus ~17 GB for the dropped Qwen3.5-4B) in `$HF_HOME` and `models/`. Check the plzen1 quota first.

## 4. Prompts

Five fixed target prompts in `prompts/targets.jsonl`:

| id | prompt |
|---|---|
| `hello` | "Hello! Who are you?" |
| `fact` | a factual question, e.g. "Explain why the sky is blue and why sunsets are red." |
| `story` | open-ended story, e.g. "Write a short story about a lighthouse keeper who finds a message in a bottle." |
| `math` | a multi-step word problem, solved step by step |
| `code` | e.g. "Write a Python function that parses an ISO 8601 date without using datetime, with tests." |

Plus `prompts/fillers.jsonl`: about 30 unrelated prompts of mixed length, used to build batches
and concurrent load in E2.

- **Same input tokens in both engines:** the chat template is rendered once with the HF tokenizer,
  and both engines receive the same token IDs (vLLM `prompt_token_ids`, llama.cpp `/completion` with
  a token-array prompt). This keeps template differences out of the comparison.
- `max_tokens=1024`, greedy unless stated.
- **Reasoning is off for every model:** Qwen chat templates are rendered with
  `enable_thinking=False`. E0 checks that no model emits a `<think>` block.

## 5. Experiments

**Repeats: 10 for every (model, experiment, config).** The reference GPU for the dense models is
**A100 (zia)**: changed from L40S on 2026-10-12, because fobos jobs did not start within the 2 h limit
while zia jobs start within minutes. The MoE model runs only on **grogu** (Blackwell)
and only in E1, E2 and E5.

### E0: Setup and smoke test
- Download the models and convert the GGUFs (CPU job).
- On each GPU type (L40S, A40, A100, Blackwell): load every model in both engines, generate one
  answer, and record the environment: GPU, driver, CUDA, engine version/commit, torch, model file
  hashes and node name.
- Check that the llama.app binary runs on sm_86, sm_89 and sm_120 (it was installed after probing
  sm_80), and that it provides `llama-server` with `--parallel`, `-fa` and the `/completion`
  endpoint. If not, build llama.cpp b11429 from source with
  `CMAKE_CUDA_ARCHITECTURES="80;86;89;120"`.

### E1: Repeatability at batch size 1 (RQ1), both engines
- **E1a, within one process:** 5 prompts × 10 repeats, requests sent one at a time.
- **E1b, across restarts:** 10 fresh engine processes, 1 pass over the 5 prompts each.
- **E1c, across nodes:** E1a on 3 different zia nodes (dense models only).
- E1a is the reference that E2–E5 are compared against (per engine, model and GPU).

### E2: Batching and concurrency (RQ2), greedy
**vLLM (main part):**
- **Batch size:** each target prompt in an offline batch of size 1, 2, 4 and 8 (the rest are
  fillers), with the target placed first and last in the batch.
- **Batch-invariant mode:** repeat the batch-size grid with `VLLM_BATCH_INVARIANT=1`, and measure
  throughput (tokens/s at batch size 8) with and without it.
- **Prefix cache:** cache off, cache on and cold, cache on and warm (same prompt resubmitted).
- **Server load:** `vllm serve` with 0 vs 4 concurrent clients sending fillers while the targets run.

**llama.cpp (small):** `llama-server --parallel 1` vs `--parallel 4` with 3 concurrent filler
clients. Default `-b/-ub`, nothing else.

### E3: Engine knobs (RQ3), batch size 1, dense models on L40S
| vLLM | llama.cpp |
|---|---|
| CUDA graphs (default) vs `enforce_eager=True` | `-fa on` vs `-fa off` |
| default attention backend vs one alternative that supports the model (chosen in E0) | |

One knob at a time, compared against E1a.

### E4: Cross-hardware (RQ4), dense models, both engines
E1a on **A100 (zia, sm_80, reference)**, **L40S (fobos or alfrid, sm_89)** and **A40 (galdor, sm_86)**. These have the most
nodes and cover two architectures plus two cards of the same generation. H100 (bee, sm_90) is
added only if nodes are free at submission time; we don't queue for it.

- Report how outputs diverge between each pair of GPU types, and whether any pair agrees exactly.
- Cross-engine (vLLM vs llama.cpp on the same GPU) comes free from the E1a data, including the MoE on
  grogu.

### E5: Seeded sampling (RQ5), batch size 1, both engines
- T ∈ {0.7, 1.0}, top-p 0.95, fixed seed (the same seed in every repeat).
- vLLM per-request `seed`; llama.cpp `seed` field in the `/completion` request.

## 6. Implementation

```
determinism/
  EXPERIMENTS.md
  REPORT.md            # final report
  prompts/             # targets.jsonl, fillers.jsonl
  configs/             # one YAML per experiment: model, engine, knob grid
  src/
    prompts.py         # renders chat templates to token IDs (shared by both engines)
    run_vllm.py        # offline LLM API and a client for `vllm serve`
    run_llama.py       # starts llama-server, drives it over HTTP
    envinfo.py         # environment fingerprint
    analyze.py         # metrics + plots vs the reference run
  jobs/
    prepare.sh         # download models + convert GGUFs (CPU job)
    run.sh             # PBS job: one (GPU type, model set, experiment list)
  results/<jobid>.tar.zst   # one archive per job (copy-back is slow for many small files)
```

- **One job per (GPU type, models)**, which runs all the experiments planned for that GPU in
  sequence. The model is loaded once per config that needs a reload.
- Output record: `{config, env, prompt_id, run_idx, token_ids, top_logprobs, text_sha}`.
- Job sizing: 1 GPU, 4 CPUs, 32 GB RAM, `scratch_local=40gb` (model copied to scratch),
  ≤ 4 h walltime.

**Budget estimate.** A full pass is 5 prompts × 10 repeats × ≤ 1024 tokens ≈ 50k generated tokens.
At batch size 1 that is about 5–10 min for the 4B model and less for the 350M one.

| Job | GPU | Content | Est. GPU-h |
|---|---|---|---|
| prepare | CPU | downloads + conversion | (CPU only) |
| E0 smoke | 4 × 1 GPU | short runs | ~1 |
| reference | fobos | E1, E2, E3, E5 for both dense models | ~3–4 |
| E1c | 2 × zia | E1a on 2 more nodes | ~1 |
| E4 | galdor, zia (+ bee) | E1a for both dense models | ~1–1.5 |
| MoE | grogu | E1, E2, E5 | ~2–3 |
| **total** | | | **~8–10 GPU-h estimated; revised 2026-10-12 to ~15 GPU-h (vLLM batch grid is ~2.5 h per model on its own)** |

- **Queue limit: 2 h.** The submit helper cancels (`qdel`) any job that hasn't started within 2 h of
  submission. In that case:
  - L40S (E4): resubmit once on another L40S cluster (alfrid) or skip;
  - E4: skip that GPU type and note it in the report;
  - MoE (grogu): the MoE part is dropped.
