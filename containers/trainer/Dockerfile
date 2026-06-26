# trainer image (CPU, mock). Runs the real personal_style env + Tinfoil judge,
# no GPU. Swap to Dockerfile.gpu (prime-rl) when the H200 lands.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

WORKDIR /app
COPY . .
RUN uv sync

ENV STATE_DIR=/state PATH="/app/.venv/bin:${PATH}"
EXPOSE 9100
CMD ["uvicorn", "trainer_service.app:app", "--host", "0.0.0.0", "--port", "9100"]
