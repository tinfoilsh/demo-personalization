"""Auth. A shared demo key gates access; the per-request encryption key IS the
identity — there is no user/account concept.

The key deterministically derives a public `adapter_id` (used as the blob filename
and the vLLM model name) and, separately, the AES key that encrypts the blob. The
two derivations are domain-separated, so the public id (a filename on disk) reveals
nothing about the decryption key. Nothing about the caller is ever stored: their
key names and decrypts their adapter, and that's the whole identity model.
"""

import hashlib
from dataclasses import dataclass

from fastapi import Header, HTTPException

from control_server import config

_ID_DOMAIN = b"tinfoil-personalization/id\x00"


@dataclass
class Principal:
    encryption_key: str

    @property
    def adapter_id(self) -> str:
        """Public id derived from the key — names the blob + the vLLM model."""
        return hashlib.sha256(_ID_DOMAIN + self.encryption_key.encode()).hexdigest()[:16]

    @property
    def adapter_name(self) -> str:
        return f"adapter-{self.adapter_id}"


def authenticate(
    x_demo_key: str = Header(...),
    x_encryption_key: str = Header(...),
) -> Principal:
    if x_demo_key != config.DEMO_KEY:
        raise HTTPException(status_code=401, detail="bad demo key")
    if len(x_encryption_key) < 8:
        raise HTTPException(status_code=401, detail="encryption key too short")
    return Principal(encryption_key=x_encryption_key)
