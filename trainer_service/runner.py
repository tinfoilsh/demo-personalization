"""Runs prime-rl training jobs. Lives in the TRAINING container (has the GPU +
prime-rl). One job == one `uv run rl @ rl.toml` subprocess, which itself
supervises rollout-inference + orchestrator + env-server + trainer.

Reads the user's corpus and writes the adapter under the shared volume, so the
control container can hot-load it into serving afterwards.

MOCK_MODE=1 (no-GPU deploys) runs the env + judge against Tinfoil instead of
`uv run rl`: real environment, real judge, no gradient step, no adapter.
"""

import asyncio
import json
import os
import statistics
from pathlib import Path

STATE_DIR = Path(os.environ.get("STATE_DIR", "./state")).resolve()
JOBS_DIR = STATE_DIR / "jobs"
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

# job_id -> {"status": ..., "adapter_path": ..., "error": ...}
_jobs: dict[str, dict] = {}


def _render_config(job_dir: Path, corpus_path: str) -> Path:
    toml_path = job_dir / "rl.toml"
    toml_path.write_text(
        RL_TOML.format(
            max_steps=MAX_STEPS,
            output_dir=job_dir.as_posix(),
            base_model=BASE_MODEL,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            env_id=ENV_ID,
            corpus_path=corpus_path,
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


async def _run_mock(job_id: str, corpus_path: str) -> None:
    """No-GPU path: run the real env + judge against Tinfoil, no gradient step."""
    from personal_style import load_environment

    from tinfoil_llm import POLICY_MODEL, policy_client

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    env = load_environment(corpus_path=corpus_path)
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
    (job_dir / "mock_result.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    # No adapter in mock — control skips the hot-load and serving stays on base.
    _jobs[job_id] = {
        "status": "ready",
        "adapter_path": None,
        "mock": True,
        "mean_reward": mean_reward,
    }


async def _run(job_id: str, corpus_path: str) -> None:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    toml_path = _render_config(job_dir, corpus_path)

    proc = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "rl",
        "@",
        toml_path.as_posix(),
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
    except FileNotFoundError as e:
        _jobs[job_id] = {"status": "failed", "error": str(e)}
        return
    _jobs[job_id] = {"status": "ready", "adapter_path": adapter_dir.as_posix()}


def start(job_id: str, corpus_path: str) -> None:
    _jobs[job_id] = {"status": "training"}
    asyncio.create_task((_run_mock if MOCK_MODE else _run)(job_id, corpus_path))


def get(job_id: str) -> dict:
    return _jobs.get(job_id, {"status": "unknown"})
