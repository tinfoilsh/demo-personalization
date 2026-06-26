"""Dirt-simple AES-GCM encryption for adapters at rest.

The disk is NOT encrypted at the volume level, so anything sensitive that must
persist — the trained LoRA adapter — is encrypted here first. The user's
per-request encryption key is the only secret; we never store it. The corpus is
never persisted at all (it's socketed to the trainer and kept in RAM).

Key = SHA-256 of the user's encryption key (32 bytes → AES-256-GCM). Each blob is
`nonce(12) || ciphertext`. Adapters are directories, so we tar them first.
"""

import hashlib
import io
import os
import tarfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _key(encryption_key: str) -> bytes:
    return hashlib.sha256(encryption_key.encode()).digest()


def encrypt_dir(src_dir: Path, dest_file: Path, encryption_key: str) -> None:
    """Tar `src_dir`, AES-GCM encrypt, write `nonce || ciphertext` to `dest_file`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(src_dir, arcname=".")
    nonce = os.urandom(12)
    ciphertext = AESGCM(_key(encryption_key)).encrypt(nonce, buf.getvalue(), None)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(nonce + ciphertext)


def decrypt_dir(src_file: Path, dest_dir: Path, encryption_key: str) -> None:
    """Reverse of `encrypt_dir`: decrypt `src_file` and extract into `dest_dir`.

    Raises cryptography.exceptions.InvalidTag if the key is wrong.
    """
    blob = src_file.read_bytes()
    nonce, ciphertext = blob[:12], blob[12:]
    data = AESGCM(_key(encryption_key)).decrypt(nonce, ciphertext, None)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        tar.extractall(dest_dir, filter="data")
