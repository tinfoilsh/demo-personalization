"""Serving mock: stands in for the multi-LoRA vLLM in no-GPU deploys.

Exposes the same surface `control_server.serving` calls — `/v1/load_lora_adapter`
(no-op, since there are no real adapters) and `/v1/chat/completions` (proxied to
the Tinfoil base model). Keeps the full three-container topology testable without
a GPU; swap this image for the real vLLM one when the H200 lands.

    uv run uvicorn serving_mock.app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI

from tinfoil_llm import POLICY_MODEL, policy_client

app = FastAPI(title="serving (mock)")


@app.post("/v1/load_lora_adapter")
def load_lora_adapter(body: dict) -> dict:
    # No real adapters in mock; accept and ignore so the control flow is unchanged.
    return {"status": "mock-noop", "lora_name": body.get("lora_name")}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict) -> dict:
    # Ignore the per-user adapter in `model`; serve the base model from Tinfoil.
    resp = await policy_client().chat.completions.create(
        model=POLICY_MODEL,
        messages=body.get("messages", []),
        temperature=body.get("temperature", 0.7),
        max_tokens=body.get("max_tokens", 512),
    )
    return resp.model_dump()
