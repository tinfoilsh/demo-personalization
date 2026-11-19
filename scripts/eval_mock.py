"""Mock-mode eval: run the personal_style environment with no GPU.

Points the policy and judge at Tinfoil-hosted models and runs the full env loop
(generate -> judge -> reward) against the built-in sample data. This exercises
everything except the gradient step (which needs the H200). The same env plugs
into prime-rl unchanged once the GPU lands.

Requires: TINFOIL_API_KEY in the environment.
Optional: N (examples), R (rollouts). Models are hardcoded to gpt-oss-120b.

    uv run python scripts/eval_mock.py
"""

import asyncio
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from personal_style import load_environment
from tinfoil_llm import judge_client, policy_client, POLICY_MODEL

load_dotenv()

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


async def main() -> None:
    if not os.environ.get("TINFOIL_API_KEY"):
        raise SystemExit("Set TINFOIL_API_KEY first (export TINFOIL_API_KEY=...).")

    n = int(os.environ.get("N", "3"))
    r = int(os.environ.get("R", "2"))

    # The judge is a verified Tinfoil client.
    env = load_environment(judge_client=await judge_client())
    client = await policy_client()

    print(f"policy model: {POLICY_MODEL}   examples: {n}   rollouts/example: {r}")
    results = await env.evaluate(
        client=client,
        model=POLICY_MODEL,
        num_examples=n,
        rollouts_per_example=r,
        save_results=False,
    )

    rewards = [o["reward"] for o in results.get("outputs", []) if o.get("reward") is not None]
    if rewards:
        print(f"\nmean reward: {statistics.mean(rewards):.3f}   (n={len(rewards)})")
        print(f"rewards: {[round(x, 3) for x in rewards]}")

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
