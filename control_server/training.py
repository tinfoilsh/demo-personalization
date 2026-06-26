"""Drive a per-user training job from the control container.

Control owns the user's key and the serving handoff; the actual `uv run rl` lives
in the trainer service (separate GPU container). So this module:
  1. sockets the user's corpus to the trainer (can't persist to unencrypted disk),
  2. polls until the job finishes,
  3. for a real job, decrypts the resulting adapter blob into RAM and loads it
     into serving. (Mock jobs produce no adapter — serving stays on the base.)
"""

import asyncio
import uuid

import httpx

from control_server import config, registry, serving
from control_server.auth import Principal

POLL_INTERVAL_S = 10


async def _drive(principal: Principal, job_id: str, documents: list[str]) -> None:
    async with httpx.AsyncClient(base_url=config.TRAINER_URL, timeout=60) as client:
        await client.post(
            "/jobs",
            json={
                "job_id": job_id,
                "documents": documents,
                "encryption_key": principal.encryption_key,
            },
        )
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            job = (await client.get(f"/jobs/{job_id}")).json()
            if job["status"] in ("training", "unknown"):
                continue
            if job["status"] != "ready":
                registry.set_status(principal.user_id, "failed", error=job.get("error"))
                return
            break

    blob = job.get("adapter_blob")
    try:
        if blob:  # real job — decrypt into RAM and load; mock has no adapter
            await load_adapter(principal, blob)
    except Exception as e:  # noqa: BLE001 — surface any handoff failure to the user
        registry.set_status(principal.user_id, "failed", error=str(e))
        return
    registry.set_status(
        principal.user_id, "ready", adapter_blob=blob, loaded=bool(blob)
    )


async def load_adapter(principal: Principal, blob: str) -> None:
    """Decrypt the adapter blob into RAM and register it with serving.

    Needs the user's key, so it runs at train-finish and again on first chat after
    a serving restart (RAM is gone, the encrypted blob on disk survives).
    """
    from pathlib import Path

    from cryptobox import decrypt_dir

    dest = config.DECRYPT_DIR / principal.adapter_name
    decrypt_dir(Path(blob), dest, principal.encryption_key)
    await serving.hot_load(principal.adapter_name, dest.as_posix())


def start_training(principal: Principal, documents: list[str]) -> str:
    job_id = uuid.uuid4().hex[:12]
    registry.set_status(principal.user_id, "training", job_id=job_id)
    asyncio.create_task(_drive(principal, job_id, documents))
    return job_id
