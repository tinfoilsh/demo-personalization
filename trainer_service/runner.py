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
REPO_DIR = Path(__file__).resolve().parent.parent

MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"
MOCK_N = int(os.environ.get("MOCK_N", "3"))
MOCK_R = int(os.environ.get("MOCK_R", "2"))

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

[trainer.optim]
lr = 1e-5

[orchestrator]
batch_size = 128
group_size = 8

[orchestrator.train.sampling]
max_completion_tokens = 768

[[orchestrator.train.env]]
id = "{env_id}"
name = "{env_id}"
args = {{ corpus_path = "{corpus_path}" }}

[inference]
"""

# job_id -> {"status": ..., "adapter_blob": ..., "error": ..., "mock": ..., "mean_reward": ...}
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
        )
    )
    return toml_path


def _final_adapter_dir(job_dir: Path) -> Path:
    """prime-rl writes HF-compatible adapters to <output_dir>/weights/step_*."""
    steps = sorted(
        (job_dir / "weights").glob("step_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not steps:
        raise FileNotFoundError(f"no weights/step_* under {job_dir}")
    return steps[-1]


async def _run_mock(job_id: str, documents: list[str], encryption_key: str) -> None:
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
    _jobs[job_id] = {
        "status": "ready",
        "adapter_blob": None,  # no adapter in mock — serving stays on base
        "mock": True,
        "mean_reward": mean_reward,
    }


async def _run(job_id: str, documents: list[str], encryption_key: str) -> None:
    job_dir = RAM_DIR / job_id  # RAM: corpus, toml, plaintext weights
    job_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = _write_corpus_ram(job_dir, documents)
    toml_path = _render_config(job_dir, corpus_path)

    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "rl", "@", toml_path.as_posix(),
        cwd=REPO_DIR,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    rc = await proc.wait()
    if rc != 0:
        _jobs[job_id] = {"status": "failed", "error": f"rl exited {rc}"}
        return
    try:
        adapter_dir = _final_adapter_dir(job_dir)
        # Encrypt the plaintext adapter (in RAM) to a single blob on the
        # persistent volume; control decrypts it into RAM for serving.
        blob = ADAPTERS_DIR / f"{job_id}.enc"
        encrypt_dir(adapter_dir, blob, encryption_key)
    except FileNotFoundError as e:
        _jobs[job_id] = {"status": "failed", "error": str(e)}
        return
    _jobs[job_id] = {"status": "ready", "adapter_blob": blob.as_posix()}


def start(job_id: str, documents: list[str], encryption_key: str) -> None:
    _jobs[job_id] = {"status": "training"}
    runner = _run_mock if MOCK_MODE else _run
    asyncio.create_task(runner(job_id, documents, encryption_key))


def get(job_id: str) -> dict:
    return _jobs.get(job_id, {"status": "unknown"})
