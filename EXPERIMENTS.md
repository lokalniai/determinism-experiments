# Determinism in LLM inference: experiment plan

Status: **v4, 2026-10-15; simplified to four experiments (E1–E4).** Results in [REPORT.md](REPORT.md).
Comment inline (`> ZK: ...`) or edit freely.

v3 (2026-10-12) had more settings: restarts and separate jobs at batch size 1, prefix caching,
`vllm serve` under load, `llama-server --parallel 1`, the target last in a batch, and engine knobs
(CUDA graphs, attention backend, flash attention). Those were run but are left out of the analysis
now; their raw records are still in `results/*.tar.gz`. Small scale, single GPU (no tensor
parallelism), output is a report in Markdown and the [viewer](https://lokalniai.github.io/determinism-experiments/).

## 1. Research questions

- **RQ1 (repeatability, E1):** Is decoding at batch size 1 bit-identical across repeats in one
  process, greedy and with seeded sampling? (vLLM and llama.cpp)
- **RQ2 (batching, E2):** Does the output for a fixed prompt change when other sequences share the
  forward pass? (vLLM offline batches, llama.cpp parallel slots)
- **RQ3 (enforcing determinism, E3):** Does vLLM's batch-invariant mode remove the batching effect,
  and what does it cost in throughput?
- **RQ4 (hardware, E4):** Does the same setting give the same output on a different GPU (L40S vs A100)?

## 2. What we measure

Every (prompt, repeat) record stores the generated token IDs and the top-20 logprobs at each step.
Compared against a reference run (§5):

| Metric | Meaning |
|---|---|
| match | fraction of runs whose token sequence equals the reference |
| outputs (uniq) | distinct completions per prompt across the 10 repeats (1 = deterministic) |
| div@ | median token position of the first mismatch among runs that differ |
| gap | median gap between the reference's top-2 logprobs at that position (whether outputs split on near-ties) |

## 3. Models

All ungated on Hugging Face. Both engines get weights converted from the **same checkpoint**, so
engine differences come from kernels, not from different quantizations.

| Role | vLLM | llama.cpp | Size | GPU |
|---|---|---|---|---|
| tiny dense | `LiquidAI/LFM2.5-350M` (BF16) | BF16 GGUF, converted by us | ~0.7 GB | A100 (zia), L40S in E4 |
| mid dense | `Qwen/Qwen3-4B-Instruct-2507` (BF16) | BF16 GGUF, converted by us | ~8 GB | A100 (zia), L40S in E4 |
| MoE | `nvidia/Qwen3.6-35B-A3B-NVFP4` | NVFP4 GGUF, converted by us from the same checkpoint | ~23 GB | RTX PRO 6000 Blackwell (grogu) only |

- GGUFs are converted with `convert_hf_to_gguf.py` from llama.cpp b11429 (the binary's commit);
  `models/manifest.tsv` has the revisions and hashes.
- MoE precision: the NVIDIA checkpoint is mixed. Routed and shared experts and `lm_head` are W4A16
  NVFP4, and the GGUF repacks exactly these tensors. The attention projections are FP8 in the
  checkpoint; the converter dequantized them to BF16. vLLM runs them in FP8 (`modelopt_mixed`,
  Marlin W4A16 for the experts, CUTLASS FP8 GEMM), llama.cpp in BF16.
- Architectures: Qwen3.6 mixes full attention with Gated DeltaNet linear attention (vLLM loads it
  with `language_model_only`); LFM2.5 mixes convolution and attention layers; Qwen3-4B is a
  standard transformer.
- **Batch-invariant mode (E3) needs standard attention.** vLLM 0.31 supports
  `VLLM_BATCH_INVARIANT=1` only with the FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION and MLA backends.
  It refuses LFM2's short-conv layers (`SHORT_CONV_ATTN`), and for the MoE it selects
  `CutlassNvFp4LinearKernel`, which "does not support W4A16" (its GDN layers would be refused too).
  So E3 exists only for Qwen3-4B, which replaced the hybrid `Qwen/Qwen3.5-4B` on 2026-10-12 for
  this reason.

## 4. Prompts

Five fixed target prompts in `prompts/targets.jsonl`: `hello` ("Hello! Who are you?"), `fact` (why
the sky is blue and sunsets red), `story` (a lighthouse keeper finds a message in a bottle), `math`
(a multi-step word problem), `code` (an ISO 8601 parser without `datetime`, with tests). Plus
`prompts/fillers.jsonl`: 30 unrelated prompts of mixed length, which share the batch in E2/E3.

- **Same input tokens in both engines:** the chat template is rendered once with the HF tokenizer
  (`src/render_prompts.py`), and both engines receive those token IDs (vLLM `prompt_token_ids`,
  llama.cpp `/completion` with a token array).
- `max_tokens=1024`, greedy unless stated, reasoning off (`enable_thinking=False`).

## 5. Experiments

**10 repeats per (setting, model, prompt).** Reference GPU: A100 (zia) for the dense models,
Blackwell (grogu) for the MoE.

| | Setting | vLLM | llama.cpp | Reference |
|---|---|---|---|---|
| **E1** | batch size 1, one process, requests one at a time | `e1` greedy, `e1_t07`, `e1_t10` | the same | repeat 0 of the same setting |
| **E2** | batching, greedy | `e2_bs{2,4,8}`: offline batch of N, target first, N−1 random fillers (new per repeat) | `e2_np{2,4,8}`: `llama-server --parallel N`, N−1 clients sending fillers concurrently | E1 greedy, repeat 0 |
| **E3** | batch-invariant mode, greedy | `e3_bs{1,2,4,8}`: E2's batches with `VLLM_BATCH_INVARIANT=1` (Qwen3-4B only) | – | `e3_bs1`, repeat 0 |
| **E4** | other GPU | `e4`: E1 greedy on L40S (alfrid), dense models | the same | E1 greedy on A100, repeat 0 |

- Sampling (E1): T ∈ {0.7, 1.0}, top-p 0.95, the same seed (1234) in every repeat; vLLM per-request
  `seed`, llama.cpp `seed` in the `/completion` request.
- E3 compares against its own batch-1 run: the invariant kernels are a different computation, so
  the question is whether batching still changes the output, not whether it equals default vLLM.
- E3 throughput: whole-batch tokens/s, compared with the default (E2) batches.
- E4 is greedy only; the MoE is not part of it (Blackwell only).

## 6. Implementation

```
determinism/
  EXPERIMENTS.md, REPORT.md
  prompts/             targets.jsonl, fillers.jsonl, rendered/ (token IDs per model)
  src/
    render_prompts.py  chat templates -> token IDs (shared by both engines)
    run_vllm.py        offline LLM API, one worker process per (model, setting)
    run_llama.py       starts llama-server per setting, drives it over HTTP
    common.py          models, environment fingerprint, record writer
    analyze.py         metrics vs the reference -> results/summary.md
    site_data.py       reduction for the viewer -> results/site_data.json.gz
    build_site.py      the viewer (static HTML)
  jobs/
    prepare.sh         download models + convert GGUFs (CPU job)
    build_llama.sh     llama.cpp b11429 for sm_80/86/89/120a
    run.sh             PBS job: one (GPU, engines, models, settings)
    analyze.sh         analyze.py + site_data.py (CPU job; parses ~1.5 GB of records)
  results/<jobid>.<engines>.<cluster>.tar.gz   one archive per job
```

- One job per (GPU type, models). Record: `{exp, engine, model, config, prompt_id, rep, token_ids,
  top_logprobs, text, ...}`.
- Job sizing: 1 GPU, 4 CPUs, 32 GB RAM, `scratch_local=40gb` (vLLM venv unpacked to scratch), ≤ 1 h.
- The runners use the setting names above. Jobs before 2026-10-15 used older names, which
  `src/analyze.py` maps: `e1a` → `e1`, `e5_t07`/`e5_t10` → `e1_t07`/`e1_t10`, `e2bi_bs{N}` →
  `e3_bs{N}`; `e1a` on L40S → `e4`.

**Runs** (all on 2026-10-12 unless stated):

| Job | GPU | Content |
|---|---|---|
| 24602570, 24603114 | A100 | vLLM E1, LFM2.5 / Qwen3-4B |
| 24602570, 24603116 | A100 | vLLM E2 (LFM2.5 / Qwen3-4B), E3 (Qwen3-4B) |
| 24603117 | A100 | llama.cpp E1, E2 `--parallel 4` |
| 24690400 (2026-10-15) | A100 | llama.cpp E2 `--parallel 2` and `8` |
| 24602683, 24602684 | Blackwell | vLLM E1, E2 (MoE) |
| 24602769 | Blackwell | llama.cpp E1, E2 `--parallel 4` (MoE) |
| 24690401 (2026-10-15) | Blackwell | llama.cpp E2 `--parallel 2` and `8` (MoE) |
| 24604441 | L40S | E4, both engines |

Queue limit: a helper (`jobs/watchdog.sh`) cancels a job still queued 2 h after submission. A40
and H100 were dropped from E4 under this rule.
