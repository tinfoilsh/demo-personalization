"""In-memory user -> adapter state.

The one bit of state the control server must keep: which user owns which adapter
and whether it's trained yet, so chat requests route to the right `model`.

In-memory for the demo. In prod this persists to the encrypted volume so it
survives a restart (the adapters themselves already live there).
"""

from dataclasses import dataclass
from typing import Literal

Status = Literal["none", "training", "ready", "failed"]


@dataclass
class UserState:
    user_id: str
    adapter_name: str
    status: Status = "none"
    job_id: str | None = None
    adapter_path: str | None = None
    error: str | None = None


_users: dict[str, UserState] = {}


def get(user_id: str, adapter_name: str) -> UserState:
    if user_id not in _users:
        _users[user_id] = UserState(user_id=user_id, adapter_name=adapter_name)
    return _users[user_id]


def set_status(user_id: str, status: Status, **fields) -> None:
    state = _users[user_id]
    state.status = status
    for k, v in fields.items():
        setattr(state, k, v)
