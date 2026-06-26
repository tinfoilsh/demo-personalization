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

BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
# prime-rl picks a chat-template renderer per model; Llama-3.1-8B isn't in its
# MODEL_RENDERER_MAP, so name it explicitly (template-identical to the mapped
# Llama-3.2 Instruct models).
RENDERER = os.environ.get("RENDERER", "llama-3")
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
INFER_GPU_UTIL = os.environ.get("INFER_GPU_UTIL", "0.4")

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

# We never resume and the RAM-backed tmpfs is tight, so keep nothing but the LoRA
# adapter. weights_only drops the optimizer/dataloader state; skip_gather avoids
# materialising the full base model in RAM at checkpoint time (a ~16 GB gather for
# the 8B that OOMs the host).
[trainer.ckpt]
weights_only = true
skip_gather_master_weights = true

# Save only the LoRA adapter (PEFT format), never a full-model copy. The trained
# adapter is broadcast to <output_dir>/run_*/broadcasts/step_N each step;
# _final_adapter_dir reads the last one.
[trainer.ckpt.weights]
save_adapter_separately = true
save_sharded = false

[trainer.optim]
lr = 1e-5

[orchestrator]
batch_size = 128
group_size = 8

[orchestrator.renderer]
name = "{renderer}"

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
            renderer=RENDERER,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            env_id=ENV_ID,
            corpus_path=corpus_path.as_posix(),
            infer_gpu_util=INFER_GPU_UTIL,
        )
    )
    return toml_path


def _final_adapter_dir(job_dir: Path) -> Path:
    """prime-rl broadcasts the PEFT LoRA adapter (adapter_config.json +
    adapter_model.safetensors) to <output_dir>/run_*/broadcasts/step_N each step.
    With skip_gather_master_weights the trainer writes no weights/ checkpoint (that
    would gather the full base model and OOM the RAM-backed host), so the last
    broadcast IS the final trained adapter — that's what vLLM loads."""
    steps = sorted(
        (
            p
            for p in job_dir.glob("**/broadcasts/step_*")
            if (p / "adapter_config.json").is_file()
        ),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not steps:
        raise FileNotFoundError(f"no broadcasts/step_*/adapter_config.json under {job_dir}")
    return steps[-1]


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
    # Llama is gated; prod provides the token as LLAMA_HF_TOKEN, but HF reads
    # HF_TOKEN — bridge it for the `uv run rl` model download.
    if env.get("LLAMA_HF_TOKEN"):
        env["HF_TOKEN"] = env["LLAMA_HF_TOKEN"]
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
