"""Runs prime-rl training jobs. Lives in the TRAINING container (has the GPU +
prime-rl). One job == one `uv run rl @ rl.toml` subprocess, which itself
supervises rollout-inference + orchestrator + env-server + trainer.

Data handling (disk is NOT volume-encrypted):
  - corpus: socketed in from control, written only to RAM (tmpfs), never to the
    persistent volume. prime-rl's env reads it as a file from there.
  - rl.toml + prime-rl outputs: also under RAM, since plaintext weights are
    sensitive and must not persist.
  - adapter: encrypted (AES-GCM, user's key) and written to the persistent volume
    as a single blob. Control later decrypts it into RAM for serving.

MOCK_MODE=1 (no-GPU deploys) runs the env + judge against Tinfoil instead of
`uv run rl`: real environment, real judge, no gradient step, no adapter, nothing
on disk.
"""

import asyncio
import json
import os
import statistics
from pathlib import Path

from cryptobox import encrypt_dir

# RAM (tmpfs) for everything plaintext-and-sensitive: corpus, rl outputs.
RAM_DIR = Path(os.environ.get("RAM_DIR", "/dev/shm/personalization")).resolve()
# Persistent shared volume — only ever holds ENCRYPTED adapter blobs.
STATE_DIR = Path(os.environ.get("STATE_DIR", "./state")).resolve()
ADAPTERS_DIR = STATE_DIR / "adapters"

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50"))
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
ENV_ID = os.environ.get("ENV_ID", "personal-style")
# Working dir for `uv run rl`: locally the repo, in the GPU image prime-rl's own
# uv project (/app), where the `rl` entrypoint and its venv live.
RL_PROJECT_DIR = Path(
    os.environ.get("RL_PROJECT_DIR", Path(__file__).resolve().parent.parent)
)

MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"
MOCK_N = int(os.environ.get("MOCK_N", "3"))
MOCK_R = int(os.environ.get("MOCK_R", "2"))

# VRAM fraction for the rollout inference server. Capped (< serving + trainer
# must all fit on the single GPU); tune live. See _run for the colocation env.
INFER_GPU_UTIL = os.environ.get("INFER_GPU_UTIL", "0.2")

RL_TOML = """\
max_steps = {max_steps}
seq_len = 2048
output_dir = "{output_dir}"

[ckpt]

[model]
name = "{base_model}"

[trainer.model.lora]
rank = {lora_rank}
alpha = {lora_alpha}

# Skip the full training checkpoint (optimizer + dataloader state, ~17 GB for a
# 4B model). We never resume, and the RAM-backed tmpfs OOMs writing it; we only
# need the weight checkpoint's adapter.
[trainer.ckpt]
weights_only = true

# Write the LoRA adapter on its own (weights/step_N/lora_adapters, PEFT format)
# so vLLM can load just the adapter instead of a merged full-model checkpoint.
[trainer.ckpt.weights]
save_adapter_separately = true

[trainer.optim]
lr = 1e-5

[orchestrator]
batch_size = 128
group_size = 8

# Rollouts must run against the adapter being trained, not the base model.
[orchestrator.model.lora]
name = "{env_id}"

[orchestrator.train.sampling]
max_completion_tokens = 768

[[orchestrator.train.env]]
id = "{env_id}"
name = "{env_id}"
args = {{ corpus_path = "{corpus_path}" }}

# Rollout inference server. Colocated on the one GPU (see _run), so its memory
# is capped to leave room for the FSDP trainer and the serving container.
[inference]
enable_lora = true
gpu_memory_utilization = {infer_gpu_util}

[inference.model]
max_model_len = 2048
"""

# adapter_id -> {"status": ..., "adapter_blob": ..., "error": ..., "mock": ..., "mean_reward": ...}
_jobs: dict[str, dict] = {}


def _write_corpus_ram(job_dir: Path, documents: list[str]) -> Path:
    """Write the corpus to RAM (tmpfs) so prime-rl's env can read it as a file
    without it ever touching the persistent disk."""
    corpus_path = job_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc}) + "\n")
    return corpus_path


def _render_config(job_dir: Path, corpus_path: Path) -> Path:
    toml_path = job_dir / "rl.toml"
    toml_path.write_text(
        RL_TOML.format(
            max_steps=MAX_STEPS,
            output_dir=job_dir.as_posix(),
            base_model=BASE_MODEL,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            env_id=ENV_ID,
            corpus_path=corpus_path.as_posix(),
            infer_gpu_util=INFER_GPU_UTIL,
        )
    )
    return toml_path


def _final_adapter_dir(job_dir: Path) -> Path:
    """With save_adapter_separately, prime-rl writes the PEFT LoRA adapter to
    <output_dir>/weights/step_N/lora_adapters. That dir (adapter_config.json +
    adapter weights) is what vLLM loads — not the full weights/step_N tree."""
    steps = sorted(
        (job_dir / "weights").glob("step_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not steps:
        raise FileNotFoundError(f"no weights/step_* under {job_dir}")
    adapter_dir = steps[-1] / "lora_adapters"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"no lora_adapters under {steps[-1]}")
    return adapter_dir


async def _run_mock(adapter_id: str, documents: list[str], encryption_key: str) -> None:
    """No-GPU path: real env + judge against Tinfoil, no gradient step, no adapter,
    nothing on disk (corpus stays in memory)."""
    from personal_style import load_environment

    from tinfoil_llm import POLICY_MODEL, policy_client

    env = load_environment(user_texts=documents)
    results = await env.evaluate(
        client=policy_client(),
        model=POLICY_MODEL,
        num_examples=MOCK_N,
        rollouts_per_example=MOCK_R,
        save_results=False,
    )
    rewards = [
        o["reward"] for o in results.get("outputs", []) if o.get("reward") is not None
    ]
    mean_reward = statistics.mean(rewards) if rewards else None
    _jobs[adapter_id] = {
        "status": "ready",
        "adapter_blob": None,  # no adapter in mock — serving stays on base
        "mock": True,
        "mean_reward": mean_reward,
    }


async def _run(adapter_id: str, documents: list[str], encryption_key: str) -> None:
    job_dir = RAM_DIR / adapter_id  # RAM: corpus, toml, plaintext weights
    job_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = _write_corpus_ram(job_dir, documents)
    toml_path = _render_config(job_dir, corpus_path)

    # Colocate the rollout inference server and the FSDP trainer on the single
    # GPU. `rl` places one subprocess per entry in CUDA_VISIBLE_DEVICES, so
    # listing device 0 twice puts both on GPU 0 (filesystem weight-broadcast,
    # the default, needs no second GPU). The serving vLLM shares the same GPU.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,0"}
    # rl's full output goes to a log file in the job dir (RAM) so failures are
    # debuggable — `tail` it at {RAM_DIR}/{adapter_id}/rl.log.
    log_path = job_dir / "rl.log"
    with open(log_path, "w") as logf:
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "rl",
            "@",
            toml_path.as_posix(),
            cwd=RL_PROJECT_DIR,
            env=env,
            stdout=logf,
            stderr=asyncio.subprocess.STDOUT,
        )
        rc = await proc.wait()
    if rc != 0:
        tail = ""
        try:
            tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-30:])
        except Exception:  # noqa: BLE001
            pass
        _jobs[adapter_id] = {
            "status": "failed",
            "error": f"rl exited {rc}",
            "log_tail": tail,
        }
        return
    try:
        adapter_dir = _final_adapter_dir(job_dir)
        # Encrypt the plaintext adapter (in RAM) to a blob on the persistent
        # volume, named by adapter_id; control decrypts it into RAM for serving.
        blob = ADAPTERS_DIR / f"{adapter_id}.enc"
        encrypt_dir(adapter_dir, blob, encryption_key)
    except FileNotFoundError as e:
        _jobs[adapter_id] = {"status": "failed", "error": str(e)}
        return
    _jobs[adapter_id] = {"status": "ready", "adapter_blob": blob.as_posix()}


async def _safe_run(adapter_id: str, documents: list[str], encryption_key: str) -> None:
    """Run the job, marking it failed on ANY exception so it never hangs at
    'training' (a bare create_task swallows exceptions otherwise)."""
    fn = _run_mock if MOCK_MODE else _run
    try:
        await fn(adapter_id, documents, encryption_key)
    except Exception as e:  # noqa: BLE001 — surface every failure to the client
        _jobs[adapter_id] = {"status": "failed", "error": repr(e)}


def start(adapter_id: str, documents: list[str], encryption_key: str) -> None:
    _jobs[adapter_id] = {"status": "training"}
    asyncio.create_task(_safe_run(adapter_id, documents, encryption_key))


def get(adapter_id: str) -> dict:
    return _jobs.get(adapter_id, {"status": "unknown"})
