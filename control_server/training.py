"""Drive a per-key training job from the control container.

The user's key is the identity: it names the adapter (adapter_id) and decrypts its
blob. Control sockets the corpus to the trainer (never persisted), polls until the
job finishes, then — for a real job — decrypts the adapter into RAM and loads it
into serving. Mock jobs produce no adapter, so serving stays on the base model.
"""

import asyncio

import httpx

from control_server import config, registry, serving
from control_server.auth import Principal

POLL_INTERVAL_S = 10


async def _drive(principal: Principal, documents: list[str]) -> None:
    aid = principal.adapter_id
    async with httpx.AsyncClient(base_url=config.TRAINER_URL, timeout=60) as client:
        await client.post("/jobs", json={
            "adapter_id": aid,
            "documents": documents,
            "encryption_key": principal.encryption_key,
        })
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            job = (await client.get(f"/jobs/{aid}")).json()
            if job["status"] in ("training", "unknown"):
                continue
            if job["status"] != "ready":
                registry.mark(aid, status="failed", error=job.get("error"))
                return
            break

    try:
        if job.get("adapter_blob"):  # real job — decrypt + load now (warm first chat)
            await load_adapter(principal)
    except Exception as e:  # noqa: BLE001 — surface any handoff failure to the user
        registry.mark(aid, status="failed", error=str(e))
        return
    registry.mark(aid, status="ready", mean_reward=job.get("mean_reward"))


async def load_adapter(principal: Principal) -> None:
    """Decrypt this key's adapter blob into RAM and register it with serving.

    Needs the user's key, so it runs at train-finish and again on first chat after
    a serving restart (RAM is gone, the encrypted blob on disk survives).
    """
    from cryptobox import decrypt_dir

    dest = config.DECRYPT_DIR / principal.adapter_name
    decrypt_dir(registry.blob_path(principal.adapter_id), dest, principal.encryption_key)
    await serving.hot_load(principal.adapter_name, dest.as_posix())


def start_training(principal: Principal, documents: list[str]) -> None:
    registry.mark(principal.adapter_id, status="training")
    asyncio.create_task(_drive(principal, documents))
