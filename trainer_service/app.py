"""Trainer service: minimal job API around prime-rl. Runs in the TRAINING
container. The control server is its only client. Jobs are keyed by adapter_id
(derived from the user's key by control), which also names the encrypted blob.

    uv run uvicorn trainer_service.app:app --host 0.0.0.0 --port 9100
"""

from fastapi import FastAPI
from pydantic import BaseModel

from trainer_service import runner

app = FastAPI(title="prime-rl trainer service")


class JobRequest(BaseModel):
    adapter_id: str
    documents: list[str]   # socketed in, never written to persistent disk
    encryption_key: str    # encrypts the resulting adapter at rest (real mode)


@app.post("/jobs")
def create_job(req: JobRequest) -> dict:
    runner.start(req.adapter_id, req.documents, req.encryption_key)
    return {"adapter_id": req.adapter_id, "status": "training"}


@app.get("/jobs/{adapter_id}")
def job_status(adapter_id: str) -> dict:
    return {"adapter_id": adapter_id, **runner.get(adapter_id)}
