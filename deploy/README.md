# Deploy: three containers

```
                       ┌───────────────────────────── CVM (one H200) ─────────────────────────────┐
   user ──HTTPS──▶  control (no GPU, long-lived)                                                    
                       │  POST /train  ──▶ trainer service ──▶ `uv run rl` (transient job)          
                       │  POST /v1/chat ─▶ serving vLLM                                              
                       │                                                                            
                       │   trainer (GPU 0) ── rollout vLLM + orchestrator + env-server + trainer    
                       │   serving (GPU 0) ── long-lived multi-LoRA vLLM, OpenAI API                
                       │                                                                            
                       │   shared encrypted volume /state:  corpora (control writes)                
                       │                                     adapters (trainer writes, serving reads)
                       └────────────────────────────────────────────────────────────────────────────┘
```

| Container | GPU | Lifetime | Role |
|-----------|-----|----------|------|
| **control** | no | always-up | front door: auth, spawn training, proxy chat, user→adapter map |
| **trainer** | yes (dev 0) | always-up service, transient jobs | runs `uv run rl` per user |
| **serving** | yes (dev 0) | always-up | multi-LoRA vLLM serving finished adapters |

- **GPU sharing:** trainer and serving both reserve device 0 and share it (memory split via `--gpu-memory-utilization`, compute time-shared). Control has no GPU. See `personalization.md` §4.
- **Adapter handoff:** the shared `/state` volume. Trainer writes `jobs/<id>/weights/step_*`; control hot-loads that path into serving (`POST /v1/load_lora_adapter`).
- **Attestation:** each image is pinned by sha256 in `tinfoil-config.yml`. The *code* is attested; the per-user adapters are private runtime state on the encrypted volume, never baked into an image.

## What's verified vs not

- ✅ Both Python services import and run; control renders the per-user `rl.toml` and writes corpora; routes wired. (`uv run` the smoke path locally.)
- ⚠️ The container builds and the actual `uv run rl` training step need the GPU box — untested here. `trainer.Dockerfile` is the right shape; pin versions when you run it.
- ⚠️ `tinfoil-config.yml` is a sketch; confirm GPU/encrypted-volume keys against a current `confidential-*` repo.

## Run locally (dev box with a GPU)

```
export TINFOIL_API_KEY=... DEMO_KEY=...
docker compose -f deploy/docker-compose.yml up --build
```

Without a GPU you can still run the control + trainer services as plain processes to exercise the API shape (training will fail at the `uv run rl` step, as expected):

```
uv run uvicorn control_server.app:app --port 9000   # terminal 1
uv run uvicorn trainer_service.app:app --port 9100  # terminal 2
```
