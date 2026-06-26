"""Thin wrapper around the Tinfoil SDK.

One place to construct verified, OpenAI-compatible Tinfoil clients and to resolve
model names. Used by the personalization environment (judge) and by eval/serving
scripts (policy). Every client here verifies the enclave attestation before use —
that's the privacy guarantee: the user's writing never leaves confidential compute.

Config via env vars:
  TINFOIL_API_KEY       API key (default "tinfoil")

Always points at the Tinfoil router (auto-selected enclave). Judge and policy
models are hardcoded to gpt-oss-120b.
"""

import os

from tinfoil import AsyncTinfoilAI


def _api_key() -> str:
    return os.environ.get("TINFOIL_API_KEY", "tinfoil")


def async_client() -> "AsyncTinfoilAI":
    """A verified Tinfoil client whose `.chat`/`.embeddings` mirror AsyncOpenAI."""
    return AsyncTinfoilAI(api_key=_api_key())


def openai_async_client():
    """The underlying verified `AsyncOpenAI` — what verifiers/JudgeRubric expects."""
    return async_client().client


# Convenience aliases by role (same client today; kept distinct so judge and
# policy can later point at different models).
judge_client = openai_async_client
policy_client = openai_async_client

JUDGE_MODEL = "gpt-oss-120b"
POLICY_MODEL = "gpt-oss-120b"
