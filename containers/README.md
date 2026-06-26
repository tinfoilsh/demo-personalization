# Containers

One image per service (each gets its own attested identity). All build from the
repo root (shared code); only the command differs.

| Service | Image                            | Dockerfile (GPU, current)                          | CPU mock (no GPU)                          |
| ------- | -------------------------------- | -------------------------------------------------- | ------------------------------------------ |
| control | `…/demo-personalization/control` | `control/Dockerfile` (never needs GPU)             | same                                       |
| trainer | `…/demo-personalization/trainer` | `trainer/Dockerfile.gpu` — prime-rl, real training | `trainer/Dockerfile` + `MOCK_MODE=1`       |
| serving | `…/demo-personalization/serving` | `serving/Dockerfile.gpu` — vLLM multi-LoRA         | `serving/Dockerfile` — proxy to base model |

Everything colocates on one GPU (`gpus: 1`): `uv run rl` runs the FSDP trainer and
the rollout vLLM on GPU 0 (`CUDA_VISIBLE_DEVICES=0,0`), alongside the serving
vLLM, each capping its VRAM fraction.

## Build / release

`.github/workflows/tinfoil-release.yml` (run with a version) builds all three,
writes their digests into `../tinfoil-config.yml`, tags, and dispatches
`tinfoil-release-publish.yml` to measure + attest. To fall back to the no-GPU
mock, point the trainer + serving build jobs at the plain `Dockerfile`s and drop
the GPU keys (`gpus`, `runtime`, `ipc`) from the config.

## Run locally without Docker (no GPU)

```
export TINFOIL_API_KEY=... PERSONALIZATION_DEMO_KEY=demo MOCK_MODE=1
uv run uvicorn trainer_service.app:app --port 9100 --workers 1   # terminal 1
uv run uvicorn serving_mock.app:app   --port 8000 --workers 1   # terminal 2
uv run uvicorn control_server.app:app --port 9000 --workers 1   # terminal 3
# then: PERSONALIZATION_DEMO_KEY=demo ENC_KEY=my-secret-key ./scripts/smoke_live.sh
```
