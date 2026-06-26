# serving image (CPU, mock). Proxies chat to the Tinfoil base model and no-ops
# adapter loads. Swap to Dockerfile.gpu (vLLM multi-LoRA) when the H200 lands.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

WORKDIR /app
COPY . .
RUN uv sync

ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8000
CMD ["uvicorn", "serving_mock.app:app", "--host", "0.0.0.0", "--port", "8000"]
