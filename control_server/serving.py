"""Talk to the long-lived serving vLLM (separate container, multi-LoRA).

Two operations: hot-load a freshly trained adapter, and proxy a chat request to
the right adapter. The serving vLLM is started once with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True --enable-lora` so adapters can be added
without a restart.
"""

import httpx

from control_server import config


async def hot_load(adapter_name: str, adapter_path: str) -> None:
    """Register a trained adapter into the running serving engine."""
    async with httpx.AsyncClient(base_url=config.SERVING_URL, timeout=60) as client:
        resp = await client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": adapter_name, "lora_path": adapter_path},
        )
        # vLLM returns 200 on success; it's idempotent-ish but errors if the path
        # is missing, so surface that.
        resp.raise_for_status()


async def chat(adapter_name: str, body: dict) -> dict:
    """Proxy an OpenAI chat request, forcing `model` to the user's adapter."""
    payload = {**body, "model": adapter_name}
    async with httpx.AsyncClient(base_url=config.SERVING_URL, timeout=120) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()
