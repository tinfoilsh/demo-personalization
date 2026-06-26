# prime-rl on a single H200: LoRA post-training → multi-tenant LoRA serving

**Goal:** Stand up prime-rl on one box with a single H200, RL-post-train a small model (≤4B) with **LoRA**, then serve the trained adapter(s) through a vLLM OpenAI-compatible endpoint — including **multi-tenant** serving where multiple per-tenant LoRA adapters share one base model in GPU memory and are routed per-request.

**This is a proof-of-concept / testing setup**, not a hardened production deployment. Production caveats are in Phase 6 and the Caveats section.

---

## Source provenance (read this first)

Every command and claim below traces to one of three places, all verified by direct fetch/search on 2026-06-23:

1. **In-repo markdown docs** — files in the prime-rl GitHub repo under `docs/*.md`, viewable at `https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/<file>.md`. These are the authoritative source. A _partial_ rendered mirror exists at `https://docs.primeintellect.ai/prime-rl/`, but the repo `docs/` tree has more files than the curated hosted nav shows (e.g. `deployment.md` and `troubleshooting.md` exist in-repo even though the README's docs index only bullets Overview/Configuration/Training/Scaling/Algorithms/Advanced/Development). **When in doubt, read the file in the repo, not the hosted nav.**
2. **In-repo example READMEs** — `examples/<name>/README.md` and the `rl.toml` beside them.
3. **External docs** — vLLM's own documentation for the serving layer.

Full URL list with verification status is in §10. Where this plan says "read the README first," it means a specific file I deliberately did _not_ hardcode the contents of — have the agent `cat` it.

**Version note:** Latest tagged release is v0.5.0 (Mar 30 2026). LoRA auto-config landed via PR #1392. Have the agent run `git log --oneline -5` and confirm `--model.experimental.lora` and `docs/deployment.md` exist on the checked-out ref before trusting this verbatim.

---

## 0. Architecture facts that drive the plan

- A prime-rl run is **three processes**: **Trainer** (FSDP2), **Inference** (vLLM-backed, OpenAI-compatible), **Orchestrator** (CPU process driving rollouts). Source: `docs/overview.md` (hosted: docs.primeintellect.ai/prime-rl/overview).
- Trainer + inference normally run on **separate** GPUs; on one GPU you **colocate** them by capping inference memory. Source: `docs/deployment.md` (verified — contains the exact single-GPU command).
- LoRA is one flag: `--model.experimental.lora` on the trainer **auto-configures LoRA across all three processes** (adapter-only weight broadcast, derives `orchestrator.lora_name` from rank/alpha, sets `max_lora_rank`, enables dynamic LoRA updating on the vLLM inference server). It also makes `uv run inference --enable-lora` set up dynamic adapter load/unload. Source: PR #1392 "Fix auto-configure LoRA" (verified).
- Checkpoints are written to `outputs/weights/step_*` and are HF-compatible (examples `hf upload` them directly). Source: `examples/wiki_search/README.md` (verified).
- "Multi-tenant" in prime-rl's own docs = many concurrent LoRA tenants sharing one trainer + inference deployment. The production serving analogue is vLLM native multi-LoRA: one base model resident, N adapters routed per-request via the `model` field. Source: `docs/advanced.md` ("Custom modeling, multimodal, LoRA, multi-tenant") + vLLM LoRA docs (both verified).

**Hardware reality check:** H200 = 141 GB. A 4B model in bf16 is ~8 GB; LoRA optimizer state is tiny. You have huge headroom — the `--gpu-memory-utilization 0.5` split in the docs is conservative (it originates from a 0.6B example on a smaller card, PR #971). On H200 both processes fit comfortably; tune empirically.

---

## 1. Prerequisites

- 1× NVIDIA H200 (Hopper). prime-rl lists H200 as supported. `nvidia-smi` working; CUDA + driver installed.
- `uv` will manage Python (repo pins 3.12).
- Optional: `HF_TOKEN` (model pulls / uploads), `WANDB_API_KEY` (run logging), `OPENAI_API_KEY` (only if you later use the wiki_search judge/embeddings — not needed for alphabet_sort).
- ~50+ GB free disk.

---

## 2. Phase 1 — Install prime-rl

Source: prime-rl README → Setup (verified).

**Quick:**

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

**Manual (more control):**

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --all-extras
```

**Optional — Flash Attention 3** (Hopper only; compiles from source, ~20–30 min). After installing it, do NOT run `uv sync --all-extras` / plain `uv run` again (they uninstall it); use `uv sync --inexact` / `uv run --no-sync`. Source: README (verified).

```bash
uv pip install "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper" --no-build-isolation
```

**Validate** (each needs the GPU). Source: README → Validate.

```bash
uv run python -V                                   # expect 3.12
uv run trainer      @ configs/debug/rl/train.toml  # RL trainer smoke test
uv run inference    @ configs/debug/infer.toml     # inference smoke test (leave running)
uv run orchestrator @ configs/debug/orch.toml      # second pane, against the running server
uv run eval         @ configs/debug/eval.toml
```

**Auth (optional):** `uv run hf auth login` (or `export HF_TOKEN=...`); `uv run wandb login` (or `export WANDB_API_KEY=...`).

---

## 3. Phase 2 — The reference example: Alphabet Sort (4B + LoRA + single GPU)

This example is purpose-built for your scenario: trains `Qwen3-4B-Instruct-2507` to sort names alphabetically, **multi-turn RL via LoRA without SFT warmup, on a single H100 in just over an hour**. Source: README → Training Examples (verified, quoted near-verbatim from the repo).

Config is a **single file**: `examples/alphabet_sort/rl.toml` (contains trainer + orchestrator + inference sections), and per PR #1392 it **already ships with LoRA enabled** (env id `primeintellect/alphabet-sort`). So you generally do NOT need to pass `--model.experimental.lora` when using this config — it's baked in.

```bash
cat examples/alphabet_sort/README.md   # read for any env-specific args / current overrides
cat examples/alphabet_sort/rl.toml     # confirm LoRA block, base model, max_lora_rank
```

(I verified the example exists and uses a single `rl.toml` with LoRA; I did NOT fetch the file's exact field values — read them here.)

---

## 4. Phase 3 — Train: trainer + inference colocated on one GPU

The example ships a single combined `rl.toml`, so use the `rl` entrypoint with that file plus the single-GPU placement flags from `docs/deployment.md` (verified):

```bash
uv run rl @ examples/alphabet_sort/rl.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --inference.gpu-memory-utilization 0.5 \
  --wandb.project prime-lora-poc \
  --wandb.name alphabet-sort-tenant-a
```

Notes:

- LoRA is already on via the config; if you start from a _non-LoRA_ config instead, add `--model.experimental.lora`. Source: PR #1392.
- `--inference.gpu-memory-utilization 0.5` is a conservative starting point. On H200 with a 4B model you can push inference higher (≈0.6–0.8) and still fit the LoRA trainer. Tune until stable without OOM. Source: `docs/deployment.md` ("tune … such that you have enough GPU memory for the RL trainer").
- For a fast loop-validation pass, cap steps low (e.g. `--trainer.max-steps 10` — confirm exact flag name against `docs/configuration.md` / `--help`) and confirm a checkpoint appears under `outputs/weights/step_*`.
- Alternative explicit form (separate config files), also from `docs/deployment.md`: `uv run rl --trainer @ ... --orchestrator @ ... --inference @ ... --trainer-gpu-ids 0 --inference-gpu-ids 0 --inference.gpu-memory-utilization 0.5`.

**Output:** adapter checkpoints in `outputs/weights/step_*`. Source: `examples/wiki_search/README.md` (verified).

---

## 5. Phase 4 — Locate / export the trained adapter

LoRA auto-config uses **adapter-only** broadcasting, so expect adapter artifacts (`adapter_config.json` + adapter weights), not a merged full model.

```bash
ls -la outputs/weights/step_<final>/      # expect adapter_config.json + adapter weights + tokenizer files
```

Optional — push to HF (same command the examples use). Source: `examples/wiki_search/README.md` (verified).

```bash
uv run hf upload <user>/Qwen3-4B-AlphabetSort-LoRA outputs/weights/step_<final>
```

**Multi-tenant compatibility gotcha:** keep the adapter a **pure LoRA layout** — do not mark `lm_head`/embeddings as fully trainable via `modules_to_save`. A pure layout (only attention/MLP projections adapted) is what guarantees the adapter is servable by vLLM's multi-LoRA feature. If your config marks layers as fully trainable, those won't multiplex. Source: kalomaze, "RL Learning with LoRA" (prime-rl LoRA practitioner write-up).

---

## 6. Phase 5 — Serve a single trained adapter (sanity check)

prime-rl's inference entrypoint serves with LoRA enabled. Source: `examples/wiki_search/README.md` (verified).

```bash
uv run inference --enable-lora --model.name <user>/Qwen3-4B-AlphabetSort-LoRA
```

Brings up an OpenAI-compatible server (default `http://localhost:8000/v1`). Hit `/v1/chat/completions` and confirm the trained behavior. (Equivalently, stock `vllm serve <base> --enable-lora` with the adapter attached — see Phase 6; prime-rl inference is vLLM underneath.)

---

## 7. Phase 6 — Multi-tenant: many adapters, one base, routed per-request

Train **2–3 separate LoRA adapters** (repeat Phase 3 with different envs/seeds → `tenant_a/b/c`), then serve them off one base via vLLM native multi-LoRA. All endpoint/flag names below verified against vLLM's LoRA docs.

### 7a. Start vLLM with LoRA + runtime updating

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   # gates dynamic load/unload; server logs a dev-only warning

vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --max-lora-rank 32 \      # >= the rank you trained with (don't over-set; wastes memory)
  --max-loras 4 \          # max adapters active in a single batch concurrently
  --max-cpu-loras 16       # total adapters kept in CPU (LRU); MUST be >= --max-loras
```

Sources: vLLM `cli/serve` + `features/lora` + `config/lora` (all verified). `max-cpu-loras` must be ≥ `max-loras`; vLLM LRU-evicts beyond it.

### 7b. Register each tenant's adapter at runtime

Source: vLLM `features/lora` → `/v1/load_lora_adapter` (verified).

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "tenant_a", "lora_path": "/abs/path/to/tenant_a"}'
# repeat for tenant_b, tenant_c ...
```

Verify registration (adapters appear alongside the base model, with `parent` set to the base):

```bash
curl http://localhost:8000/v1/models | jq .
```

Unload with `POST /v1/unload_lora_adapter {"lora_name": "tenant_a"}`.

### 7c. Route per request via the `model` field

Source: vLLM `features/lora` (verified): requests naming an adapter are processed in parallel with base-model and other-adapter requests when `max_loras` is high enough.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tenant_a", "messages": [{"role":"user","content":"..."}]}'
```

### 7d. (Cleaner) Auto-load by name with the filesystem resolver

Instead of POSTing each adapter, point vLLM at a directory; it loads any adapter on first request-by-name. Source: vLLM `features/lora` → LoRAResolver plugins (verified).

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_PLUGINS=lora_filesystem_resolver
export VLLM_LORA_RESOLVER_CACHE_DIR=/abs/path/to/adapters

vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-lora --max-lora-rank 32
```

Expected layout (verified against vLLM / multi-LoRA deployment docs):

```
/abs/path/to/adapters/
├── tenant_a/{adapter_config.json, adapter_model.safetensors}
├── tenant_b/{adapter_config.json, adapter_model.safetensors}
└── ...
```

A request with `"model": "tenant_a"` auto-loads if not resident. (HF-hub variant: `VLLM_PLUGINS=lora_hf_hub_resolver` + `VLLM_LORA_RESOLVER_HF_REPO_LIST=<comma-sep repo ids>`.)

---

## 8. Caveats & production notes

- **Dynamic `/v1/load_lora_adapter` is officially dev-only.** vLLM's own docs state it should not be used in production unless in an isolated, fully trusted environment, and the server logs "LoRA dynamic loading & unloading is enabled… This should ONLY be used for local development!" The risk in real deployments is that nothing guarantees an adapter is loaded across all replicas or survives a restart. **On your single-box / single-replica PoC this is a non-issue** — one server, no replication. For real production, prefer the resolver plugin (7d) or an external routing layer (e.g. AIBrix ModelAdapter) that tracks which replica holds which adapter. Sources: vLLM `features/lora` (verified) + RFC #12174.
- **Lock down access.** Anything that can reach `/v1/load_lora_adapter` can register adapters. Keep the server private and front it with auth before tenant traffic.
- **Pure LoRA layout required for multiplexing** (see Phase 4 gotcha). Full per-tenant fine-tunes don't share a base and can't multiplex — keep tenants on LoRA.
- **Shared base + rank ceiling.** Train all tenant adapters from the same base (`Qwen3-4B-Instruct-2507`); set server `--max-lora-rank` ≥ the largest training rank.
- **prime-rl LoRA broadcast note.** LoRA auto-config sets adapter-only broadcast; this interacts with the weight-broadcast transport. On a single box the default filesystem transport is used, so it shouldn't bite — watch for it only if you change `weight_broadcast.type`. Source: PR #1392 discussion.

---

## 9. Suggested execution order for the agent

1. Install prime-rl (Phase 1); run the debug smoke tests; confirm the GPU is visible.
2. `cat examples/alphabet_sort/README.md` and `rl.toml`; note base model, LoRA rank, env args.
3. Short LoRA training (cap steps low) colocated on GPU 0; confirm a checkpoint lands in `outputs/weights/step_*`.
4. Serve that single adapter (Phase 5); validate with a chat request.
5. Train a 2nd and 3rd adapter (vary env/seed) → distinct tenants.
6. Bring up the multi-LoRA vLLM server (Phase 6); register all adapters; demonstrate per-`model` routing returning tenant-distinct behavior.
7. Switch to the filesystem resolver (7d); document the directory layout.
8. Write a repo README capturing the exact working commands, the tuned `gpu-memory-utilization`, and the `max-loras` / `max-lora-rank` / `max-cpu-loras` values used.

---

## 10. Cited sources (with location + verification)

In-repo prime-rl docs/examples (authoritative):

- Repo + README (setup, examples, three-process architecture) — https://github.com/PrimeIntellect-ai/prime-rl — _verified, fetched_
- `docs/overview.md` (architecture) — hosted mirror: https://docs.primeintellect.ai/prime-rl/overview — _verified_
- `docs/deployment.md` (single-GPU colocation: `--trainer-gpu-ids`/`--inference-gpu-ids`/`--gpu-memory-utilization`) — https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/deployment.md — _verified, fetched (this is the file the colocation command lives in)_
- `docs/advanced.md` (LoRA, multi-tenant training) — https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/advanced.md — _existence + scope verified; contents not line-fetched_
- `examples/alphabet_sort/README.md` + `rl.toml` (4B, LoRA, single-GPU ~1hr; single combined config, LoRA pre-enabled) — https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/alphabet_sort/README.md — _existence + properties verified; field values not fetched — agent should cat_
- `examples/wiki_search/README.md` (LoRA serve command, `outputs/weights/step_*`, `hf upload`) — https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/wiki_search/README.md — _verified, fetched_
- PR #1392 "Fix auto-configure LoRA" (`--model.experimental.lora`; `--enable-lora` dynamic load/unload; adapts the two examples) — https://github.com/PrimeIntellect-ai/prime-rl/pull/1392 — _verified, fetched_
- PR #971 "Allow RL on single GPU from rl entrypoint" (origin of the colocation pattern) — https://github.com/PrimeIntellect-ai/prime-rl/pull/971 — _verified_

External — vLLM (serving layer):

- LoRA Adapters (enable-lora, `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, per-request `model` routing, resolver plugins, dev-only warning) — https://docs.vllm.ai/en/stable/features/lora/ — _verified_
- `vllm serve` CLI (`--max-loras`, `--max-lora-rank`, `--max-cpu-loras ≥ max-loras`, `--lora-target-modules`) — https://docs.vllm.ai/en/stable/cli/serve/ — _verified_
- LoRA config reference — https://docs.vllm.ai/en/latest/api/vllm/config/lora/ — _verified_
- RFC #12174 (production multi-tenant routing rationale) — https://github.com/vllm-project/vllm/issues/12174 — _referenced_

Background / context:

- kalomaze, "RL Learning with LoRA" (prime-rl LoRA details; pure-LoRA-layout → vLLM serving compatibility) — https://kalomaze.bearblog.dev/rl-lora-ddd/ — _verified_
- Prime Intellect Lab launch (their hosted multi-tenant LoRA, for comparison) — https://www.primeintellect.ai/blog/lab — _verified earlier in thread_
