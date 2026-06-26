# control image (CPU) — its own attested identity. Build context is the repo
# root so it can install the shared project; only the command differs from the
# other services.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

WORKDIR /app
COPY . .
RUN uv sync

ENV STATE_DIR=/state PATH="/app/.venv/bin:${PATH}"
EXPOSE 9000
CMD ["uvicorn", "control_server.app:app", "--host", "0.0.0.0", "--port", "9000"]
