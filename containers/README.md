# Containers

One image per service (each gets its own attested identity). All build from the
repo root (shared code); only the command differs.

| Service | Image | Now (mock, CPU) | Later (H200) |
|---------|-------|-----------------|--------------|
| control | `…/demo-personalization/control` | `control/Dockerfile` | same (control never needs GPU) |
| trainer | `…/demo-personalization/trainer` | `trainer/Dockerfile` — real env + Tinfoil judge, no GPU | `trainer/Dockerfile.gpu` — prime-rl, drop `MOCK_MODE` |
| serving | `…/demo-personalization/serving` | `serving/Dockerfile` — proxy to base model | `serving/Dockerfile.gpu` — vLLM multi-LoRA |

## Build / release

`.github/workflows/tinfoil-release.yml` (run with a version) builds all three,
writes their digests into `../tinfoil-config.yml`, tags, and dispatches
`tinfoil-release-publish.yml` to measure + attest. To move a service to GPU, point
its build job's `file:` at the `.gpu` Dockerfile and add the GPU keys to the config.

## Run locally without Docker (no GPU)

```
export TINFOIL_API_KEY=... PERSONALIZATION_DEMO_KEY=demo MOCK_MODE=1
uv run uvicorn trainer_service.app:app --port 9100 --workers 1   # terminal 1
uv run uvicorn serving_mock.app:app   --port 8000 --workers 1   # terminal 2
uv run uvicorn control_server.app:app --port 9000 --workers 1   # terminal 3
# then: PERSONALIZATION_DEMO_KEY=demo ENC_KEY=my-secret-key ./scripts/smoke_live.sh
```
