"""Control server: the long-lived front door for the demo.

Spawns per-user training jobs (transient prime-rl runs) and proxies chat to the
long-lived serving vLLM, routing each user to their own LoRA adapter. This is a
separate, always-up process from any single training run — it outlives them.

    uv run uvicorn control_server.app:app --host 0.0.0.0 --port 9000
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from control_server import config, registry, serving, training
from control_server.auth import Principal, authenticate


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    config.DECRYPT_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="tinfoil personalization control server", lifespan=lifespan)


class TrainRequest(BaseModel):
    documents: list[str]


@app.post("/train")
async def train(
    req: TrainRequest, principal: Principal = Depends(authenticate)
) -> dict:
    if not req.documents:
        raise HTTPException(status_code=400, detail="no documents")
    state = registry.get(principal.user_id, principal.adapter_name)
    if state.status == "training":
        raise HTTPException(status_code=409, detail="already training")
    job_id = training.start_training(principal, req.documents)
    return {"job_id": job_id, "status": "training", "adapter": principal.adapter_name}


@app.get("/status")
def status(principal: Principal = Depends(authenticate)) -> dict:
    state = registry.get(principal.user_id, principal.adapter_name)
    return {
        "adapter": state.adapter_name,
        "status": state.status,
        "job_id": state.job_id,
        "error": state.error,
    }


@app.post("/v1/chat/completions")
async def chat(body: dict, principal: Principal = Depends(authenticate)) -> dict:
    state = registry.get(principal.user_id, principal.adapter_name)
    if state.status != "ready":
        raise HTTPException(
            status_code=409, detail=f"adapter not ready (status={state.status})"
        )
    # If serving restarted, its RAM is gone but the encrypted blob survives —
    # re-decrypt with the user's key and reload before proxying.
    if state.adapter_blob and not state.loaded:
        await training.load_adapter(principal, state.adapter_blob)
        registry.set_status(principal.user_id, "ready", loaded=True)
    return await serving.chat(principal.adapter_name, body)
