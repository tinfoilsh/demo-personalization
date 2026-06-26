"""Job status, keyed by adapter id (which is derived from the user's key).

No persistence, no user map. The durable record of a trained adapter is its
encrypted blob on disk — named by the key-derived adapter id, decryptable only by
that key. So control restarts lose nothing that matters: disk is the source of
truth, and the in-memory dict below is just transient status for this process
lifetime (training/failed, and mock results which have no blob).
"""

from pathlib import Path

from control_server import config


def blob_path(adapter_id: str) -> Path:
    return config.ADAPTERS_DIR / f"{adapter_id}.enc"


# adapter_id -> {"status", "mean_reward", "error", ...} for THIS process lifetime
_status: dict[str, dict] = {}


def mark(adapter_id: str, **fields) -> None:
    _status[adapter_id] = {**_status.get(adapter_id, {}), **fields}


def get(adapter_id: str) -> dict:
    # A real adapter on disk is "ready" regardless of in-memory state (it survives
    # restarts); the in-memory entry covers in-flight jobs and mock results.
    if blob_path(adapter_id).exists():
        return {**_status.get(adapter_id, {}), "status": "ready"}
    return _status.get(adapter_id, {"status": "none"})
