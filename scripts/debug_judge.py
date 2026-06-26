"""Make one raw judge call and print exactly what gpt-oss-120b returns.

Shows finish_reason, the `content` (where the JSON should land), and any
`reasoning_content`, so we can see whether reasoning is eating the budget or
the model just isn't emitting JSON.

    uv run python scripts/debug_judge.py
"""

import asyncio

from dotenv import load_dotenv

from tinfoil_llm import JUDGE_MODEL, judge_client

load_dotenv()

PROMPT = (
    "Respond with ONLY a JSON object, no other text:\n"
    '{"style_match": 7, "format_penalty": 0, "match_pros": "...", '
    '"match_cons": "...", "a_shares_author": "True", "b_shares_author": "False"}'
)


async def main() -> None:
    client = judge_client()
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.25,
        max_tokens=1500,
        extra_body={"reasoning_effort": "low"},
    )
    msg = resp.choices[0].message
    print("finish_reason:", resp.choices[0].finish_reason)
    print("content:", repr(msg.content))
    print("reasoning_content:", repr(getattr(msg, "reasoning_content", None)))


if __name__ == "__main__":
    asyncio.run(main())
