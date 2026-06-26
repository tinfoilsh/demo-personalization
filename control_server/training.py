"""Drive a per-user training job from the control container.

Control owns the user data and the serving handoff; the actual `uv run rl` lives
in the trainer service (separate GPU container). So this module:
  1. writes the user's corpus to the shared volume,
  2. asks the trainer service to run a job,
  3. polls until it finishes, then hot-loads the adapter into serving.
"""

import asyncio
import json
import uuid

import httpx

from control_server import config, registry, serving
from control_server.auth import Principal

POLL_INTERVAL_S = 10


def _write_corpus(user_id: str, documents: list[str]) -> str:
    user_dir = config.DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = user_dir / "corpus.jsonl"
    # TODO(prod): encrypt at rest with the user's encryption key.
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc}) + "\n")
    return corpus_path.as_posix()


async def _drive(principal: Principal, job_id: str, corpus_path: str) -> None:
    async with httpx.AsyncClient(base_url=config.TRAINER_URL, timeout=60) as client:
        await client.post("/jobs", json={"job_id": job_id, "corpus_path": corpus_path})
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            job = (await client.get(f"/jobs/{job_id}")).json()
            if job["status"] in ("training", "unknown"):
                continue
            if job["status"] != "ready":
                registry.set_status(principal.user_id, "failed", error=job.get("error"))
                return
            break

    try:
        await serving.hot_load(principal.adapter_name, job["adapter_path"])
    except Exception as e:  # noqa: BLE001 — surface any handoff failure to the user
        registry.set_status(principal.user_id, "failed", error=str(e))
        return
    registry.set_status(principal.user_id, "ready", adapter_path=job["adapter_path"])


def start_training(principal: Principal, documents: list[str]) -> str:
    job_id = uuid.uuid4().hex[:12]
    corpus_path = _write_corpus(principal.user_id, documents)
    registry.set_status(principal.user_id, "training", job_id=job_id)
    asyncio.create_task(_drive(principal, job_id, corpus_path))
    return job_id
