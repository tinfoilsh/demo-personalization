"""Auth. A shared demo key gates access; the per-request encryption key IS the
identity — there is no user/account concept.

The demo key travels as a header (X-Demo-Key): it only gates access, so it's fine
for the Tinfoil shim/router to see it. The encryption key, by contrast, decrypts
the user's private adapter — it must never be visible to Tinfoil. EHBP encrypts
the request body but leaves headers in plaintext, so the encryption key travels
in the JSON body (read by each endpoint), not a header.

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


def authenticate(x_demo_key: str = Header(...)) -> None:
    """Validate the demo key (header). The encryption key travels in the encrypted
    request body — see principal()."""
    if x_demo_key != config.DEMO_KEY:
        raise HTTPException(status_code=401, detail="bad demo key")


def principal(encryption_key: str) -> Principal:
    """Validate the encryption key (from the EHBP-encrypted body) and build the
    Principal. Only the verified enclave ever sees this key."""
    if len(encryption_key) < 8:
        raise HTTPException(status_code=401, detail="encryption key too short")
    return Principal(encryption_key=encryption_key)
