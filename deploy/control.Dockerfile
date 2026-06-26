# Control container: long-lived front door. No GPU, no ML deps — just the web
# layer that spawns training jobs and proxies chat.
FROM python:3.12-slim

WORKDIR /app
COPY control_server ./control_server
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pydantic

ENV STATE_DIR=/state
EXPOSE 9000
CMD ["uvicorn", "control_server.app:app", "--host", "0.0.0.0", "--port", "9000"]
