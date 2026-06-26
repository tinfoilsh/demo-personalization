"""Control-server settings, all from env vars.

Paths default under ./state for local dev; in the three-container deploy STATE_DIR
is the shared (encrypted) volume mounted into control, trainer, and serving.
"""

import os
from pathlib import Path

# Shared state (corpora live here; the trainer writes adapters here too).
STATE_DIR = Path(os.environ.get("STATE_DIR", "./state")).resolve()
DATA_DIR = STATE_DIR / "users"        # per-user corpora

# The other two containers, reached over the CVM's internal network.
SERVING_URL = os.environ.get("SERVING_URL", "http://localhost:8000")  # serving vLLM
TRAINER_URL = os.environ.get("TRAINER_URL", "http://localhost:9100")  # trainer service

# Shared demo key (gates access). Per-user encryption key comes per-request.
DEMO_KEY = os.environ.get("DEMO_KEY", "demo")
