"""Dual-key auth, per the README: every request carries a shared demo key plus a
per-user encryption key.

  - demo key      → gates access to the demo (one shared secret).
  - encryption key → identifies the user AND (in prod) encrypts their corpus and
    adapter at rest. We never store it; we derive a stable user_id from it.

In production the encryption key would key the at-rest encryption of the user's
corpus and LoRA adapter on the shared volume. Here we only derive identity from
it; the crypto is a TODO marked at the storage sites.
"""

import hashlib
from dataclasses import dataclass

from fastapi import Header, HTTPException

from control_server import config


@dataclass
class Principal:
    user_id: str
    encryption_key: str

    @property
    def adapter_name(self) -> str:
        return f"user-{self.user_id}"


def authenticate(
    x_demo_key: str = Header(...),
    x_encryption_key: str = Header(...),
) -> Principal:
    if x_demo_key != config.DEMO_KEY:
        raise HTTPException(status_code=401, detail="bad demo key")
    if len(x_encryption_key) < 8:
        raise HTTPException(status_code=401, detail="encryption key too short")
    user_id = hashlib.sha256(x_encryption_key.encode()).hexdigest()[:16]
    return Principal(user_id=user_id, encryption_key=x_encryption_key)
