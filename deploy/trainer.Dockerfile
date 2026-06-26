# Trainer container: GPU. Holds prime-rl + our personal_style env + the Tinfoil
# SDK (for the in-enclave judge). The trainer_service API shells out to
# `uv run rl`, which supervises rollout-inference + orchestrator + env-server +
# trainer for one job.
#
# NOTE: untested without a GPU box. The shape is right; pin versions when you run
# it for real. prime-rl needs CUDA + Hopper for FlashAttention.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=0 STATE_DIR=/state
RUN apt-get update && apt-get install -y python3.12 python3-pip git curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
# Our project provides the env (personal_style) + the trainer_service + tinfoil.
COPY . .
# prime-rl provides the `rl` entrypoint that `uv run rl` invokes.
RUN uv add "prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl" \
 && uv sync --all-extras \
 && uv run vf-install personal_style -p environments   # register our env by id

EXPOSE 9100
CMD ["uv", "run", "uvicorn", "trainer_service.app:app", "--host", "0.0.0.0", "--port", "9100"]
