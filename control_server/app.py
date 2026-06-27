"""Control server: the long-lived front door for the demo.

Spawns per-user training jobs (transient prime-rl runs) and proxies chat to the
long-lived serving vLLM, routing each user to their own LoRA adapter. This is a
separate, always-up process from any single training run — it outlives them.

    uv run uvicorn control_server.app:app --host 0.0.0.0 --port 9000
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from control_server import config, registry, serving, training
from control_server.auth import authenticate, principal


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    config.DECRYPT_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="tinfoil personalization control server", lifespan=lifespan)

# Browser front-end calls this from another origin. Auth is via custom headers
# (x-demo-key / x-encryption-key), not cookies, so no credentials needed — which
# lets us allow any origin for the demo. Lock this down to the real origin later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    documents: list[str]
    encryption_key: str


class StatusRequest(BaseModel):
    encryption_key: str


@app.post("/train")
async def train(req: TrainRequest, _: None = Depends(authenticate)) -> dict:
    p = principal(req.encryption_key)
    if not req.documents:
        raise HTTPException(status_code=400, detail="no documents")
    if registry.get(p.adapter_id)["status"] == "training":
        raise HTTPException(status_code=409, detail="already training")
    # One GPU, one job at a time — don't queue, just tell the caller to retry.
    if registry.is_training_active():
        raise HTTPException(
            status_code=409,
            detail="another training job is in progress; this demo trains one at a time — check back in a few minutes",
        )
    training.start_training(p, req.documents)
    return {"adapter": p.adapter_name, "status": "training"}


@app.post("/status")
def status(req: StatusRequest, _: None = Depends(authenticate)) -> dict:
    p = principal(req.encryption_key)
    return {"adapter": p.adapter_name, **registry.get(p.adapter_id)}


@app.post("/v1/chat/completions")
async def chat(body: dict, _: None = Depends(authenticate)) -> dict:
    p = principal(body.pop("encryption_key", ""))
    # Demo side-by-side: `use_base: true` routes to the base model instead of the
    # caller's adapter (base is always loaded, so no readiness check or fallback).
    if body.pop("use_base", False):
        return await serving.chat(config.BASE_MODEL, body)
    if registry.get(p.adapter_id)["status"] != "ready":
        raise HTTPException(status_code=409, detail="adapter not ready")
    try:
        return await serving.chat(p.adapter_name, body)
    except httpx.HTTPStatusError as e:
        # Serving may have restarted and lost its in-RAM adapters — vLLM 404s the
        # unknown model. The encrypted blob on disk survives, so re-decrypt with the
        # user's key and retry once. (No flag to trust: we react to serving's actual
        # state, not our memory of it.)
        if registry.blob_path(p.adapter_id).exists() and e.response.status_code in (400, 404):
            await training.load_adapter(p)
            return await serving.chat(p.adapter_name, body)
        raise
