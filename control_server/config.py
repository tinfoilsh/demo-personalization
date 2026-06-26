"""Control-server settings, all from env vars.

Disk is NOT volume-encrypted, so we keep sensitive data off it: the corpus is
socketed to the trainer (never persisted), and adapters persist only as AES-GCM
blobs. Control decrypts a blob into RAM (DECRYPT_DIR) for serving to load.
"""

import os
from pathlib import Path

# Persistent shared volume — only ever holds ENCRYPTED adapter blobs.
STATE_DIR = Path(os.environ.get("STATE_DIR", "./state")).resolve()
ADAPTERS_DIR = STATE_DIR / "adapters"

# RAM (tmpfs) shared with serving — decrypted adapters live here, never on disk.
DECRYPT_DIR = Path(os.environ.get("DECRYPT_DIR", "/dev/shm/adapters")).resolve()

# The other two containers, reached over the CVM's internal network.
SERVING_URL = os.environ.get("SERVING_URL", "http://localhost:8000")  # serving vLLM
TRAINER_URL = os.environ.get("TRAINER_URL", "http://localhost:9100")  # trainer service

# Shared demo key (gates access). Per-user encryption key comes per-request.
DEMO_KEY = os.environ.get("PERSONALIZATION_DEMO_KEY", "demo")
