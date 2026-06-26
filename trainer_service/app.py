"""Trainer service: minimal job API around prime-rl. Runs in the TRAINING
container. The control server is its only client.

    uv run uvicorn trainer_service.app:app --host 0.0.0.0 --port 9100
"""

from fastapi import FastAPI
from pydantic import BaseModel

from trainer_service import runner

app = FastAPI(title="prime-rl trainer service")


class JobRequest(BaseModel):
    job_id: str
    corpus_path: str  # on the shared volume, written by control


@app.post("/jobs")
def create_job(req: JobRequest) -> dict:
    runner.start(req.job_id, req.corpus_path)
    return {"job_id": req.job_id, "status": "training"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return {"job_id": job_id, **runner.get(job_id)}
