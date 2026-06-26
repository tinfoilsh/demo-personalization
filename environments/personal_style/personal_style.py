"""personal_style — RL environment that rewards writing like a specific user.

Fork of the community environment `apaz/writing-style-matching`
(https://app.primeintellect.ai/dashboard/environments/apaz/writing-style-matching).
All credit for the original rubric design — judge-scored style match plus an
author-identification discriminator — goes to apaz.

Changes from the original:
  - Judge runs on a Tinfoil enclave (verified, OpenAI-compatible), not Anthropic,
    so the user's writing never leaves confidential compute. `judge_client` /
    `judge_model` are injectable; defaults build a verified Tinfoil client.
  - Wording generalized from "story" -> "text" so it fits emails/messages/essays.
  - `chunk_user_corpus` builds a dataset from a user's own writing.
  - Judge `response_format` dropped; we rely on the prompt + a tolerant parser +
    retry, so it runs against any OpenAI-compatible model.
"""

import json
import os
import random
from typing import Callable

import verifiers as vf
from datasets import Dataset
from jsonschema import ValidationError, validate
from verifiers import Info, State

style_adherence_prompt = """You are evaluating how closely two continuations of a piece of writing match
in terms of writing style.

Evaluate the continuations on these criteria:

1. STYLE MATCH (0 to {style_max}):
   - Vocabulary level and word choice
   - Sentence structure and length patterns
   - Tone, voice, and perspective
   - Descriptive techniques and imagery
   - Pacing and rhythm
   - Literary devices
   - Were they written by the same author
   Note: The content of the completions can and often will differ. We're
evaluating if the STYLE matches.

   Score: 0 for complete divergence, to {style_max} if the two snippets appear
to have been written by the same author.

2. FORMAT PENALTY ({penalty_max} to 0):
   - Either continuation includes meta-commentary like "Here's the
continuation:" or "I'll continue..." or "Here's a..." or "# Continuation..."
   - Either continuation includes explanations of how it wrote the text or why,
or apologies or self-justification which are not a part of the writing.
   - Either continuation includes the text snippet quoted back
   - Either continuation includes lots of repeated characters

   Score: 0 if clean (no violations), to {penalty_max} depending on severity.

3. AUTHOR IDENTIFICATION (True or False):
   - If you think continuation A is written by the same author as the text
snippet, answer "True" for "a_shares_author". Otherwise answer "False".
   - If you think continuation B is written by the same author as the text
snippet, answer "True" for "b_shares_author". Otherwise answer "False".
   - If you think both are, answer "True" for both "a_shares_author" and
"b_shares_author".
   - If you think neither are, answer "False" for both "a_shares_author" and
"b_shares_author".

Respond with ONLY a JSON object in this exact format (no other text):
{{{{
  "format_penalty": <penalty_score>
  "match_pros": <one_sentence>,
  "match_cons": <one_sentence>,
  "style_match": <style_score>,
  "a_shares_author": <"True" or "False">,
  "b_shares_author": <"True" or "False">,
}}}}

Text snippet:
{{question}}

Continuation A:
{{answer}}

Continuation B:
{{response}}
"""

style_adherence_schema = {
    "name": "style_matching_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "format_penalty": {
                "type": "number",
                "description": "Format penalty from a negative penalty_max to 0",
            },
            "match_pros": {
                "type": "string",
                "description": "Explanation of style matching strengths",
            },
            "match_cons": {
                "type": "string",
                "description": "Explanation of style matching weaknesses",
            },
            "style_match": {
                "type": "number",
                "description": "Style match score from 0 to style_max",
            },
            "a_shares_author": {
                "type": "boolean",
                "description": "True if continuation A shares the same author as the text snippet",
            },
            "b_shares_author": {
                "type": "boolean",
                "description": "True if continuation B shares the same author as the text snippet",
            },
        },
        "required": [
            "match_pros",
            "match_cons",
            "style_match",
            "a_shares_author",
            "b_shares_author",
            "format_penalty",
        ],
        "additionalProperties": False,
    },
}


def validate_and_parse_style_adherence_judge_response(
    judge_response: str, style_max: int, penalty_max: int, order_swapped: bool
) -> dict:
    """Validate and parse the judge response against the JSON schema.

    Models don't always honor constrained generation, so we extract the JSON,
    validate it, and let the caller retry on any ValueError.
    """
    if not judge_response:
        raise ValueError("Judge response is empty")

    start = judge_response.find("{")
    end = judge_response.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in judge response")

    try:
        scores = json.loads(judge_response[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in judge response: {e}")

    # Without constrained decoding the model follows the prompt literally and
    # emits the author flags as the strings "True"/"False"; coerce to bool.
    for key in ("a_shares_author", "b_shares_author"):
        if isinstance(scores.get(key), str):
            scores[key] = scores[key].strip().lower() == "true"

    try:
        validate(instance=scores, schema=style_adherence_schema["schema"])
    except ValidationError as e:
        raise ValueError(f"Judge response does not match schema: {e.message}")

    penalty = scores["format_penalty"]
    if not (penalty_max <= penalty <= 0):
        raise ValueError(
            f"format_penalty must be between {penalty_max} and 0, got {penalty}"
        )

    style_match = scores["style_match"]
    if not (0 <= style_match <= style_max):
        raise ValueError(
            f"style_match must be between 0 and {style_max}, got {style_match}"
        )

    if order_swapped:
        scores["a_correct"] = not scores["a_shares_author"]
        scores["b_correct"] = scores["b_shares_author"]
    else:
        scores["a_correct"] = scores["a_shares_author"]
        scores["b_correct"] = not scores["b_shares_author"]

    return scores


def chunk_user_corpus(
    texts: list[str], min_words: int = 80, max_pairs: int = 200
) -> Dataset:
    """Split a user's writing into (prefix, real-continuation) pairs.

    Each document is cut in half: the first half is the prompt the policy must
    continue, the second half is the user's real text the judge compares against.
    """
    snippets, continuations = [], []
    for text in texts:
        words = text.split()
        if len(words) < min_words:
            continue
        cut = len(words) // 2
        snippets.append(" ".join(words[:cut]))
        continuations.append(" ".join(words[cut:]))
        if len(snippets) >= max_pairs:
            break
    if not snippets:
        raise ValueError("No usable documents — each needs >= min_words words.")
    return Dataset.from_dict(
        {
            "story_snippet": snippets,  # generic text prefix
            "actual_continuation": continuations,  # the user's real text
        }
    )


def create_sample_dataset() -> Dataset:
    examples = [
        {
            "story_snippet": "Sarah pressed her face against the cold glass. Rain streaked down the window in little rivers, each drop racing the others to the bottom.",
            "actual_continuation": "She traced one with her finger, following its path until it merged with another and disappeared into the sill. Outside, the street was empty except for a single figure hunched beneath a black umbrella.",
        },
        {
            "story_snippet": "The diagnostic panel flashed red. Commander Chen's jaw tightened as she ran the numbers again—still showing a 73% probability of cascade failure within the next six hours.",
            "actual_continuation": "She toggled through the backup systems, calculating alternate power routing configurations. The station's survival depended on her next decision, and she had exactly ninety seconds before the next relay burned out.",
        },
        {
            "story_snippet": "Grandma's kitchen always smelled like cinnamon and old memories. I sat at the worn wooden table, its surface scarred with decades of family dinners, and watched her roll out pie dough with practiced ease.",
            "actual_continuation": "Her hands moved with a rhythm I'd seen a thousand times, pressing and folding, pressing and folding. She hummed an old tune I couldn't quite place—maybe from when she was young, maybe from before I was born.",
        },
        {
            "story_snippet": "The marketplace hummed with chaos. Merchants bellowed prices, children darted between stalls, and the air hung thick with spices, sweat, and possibility.",
            "actual_continuation": "I shouldered through the crowd, one hand on my coin purse. A vendor grabbed my sleeve, thrusting dried figs toward my face. 'Freshest in the city!' he swore, though I could see the wrinkled skins from here.",
        },
        {
            "story_snippet": "Subject 47 exhibited anomalous behavior during Trial 6. Initial response time decreased by 2.3 seconds relative to baseline, suggesting adaptation to the modified stimulus protocol.",
            "actual_continuation": "Further investigation revealed elevated cortisol markers consistent with acute stress response. Subsequent trials should incorporate longer acclimation periods to control for environmental variables. Preliminary data attached in Appendix C.",
        },
    ]
    return Dataset.from_dict(
        {
            "story_snippet": [ex["story_snippet"] for ex in examples],
            "actual_continuation": [ex["actual_continuation"] for ex in examples],
        }
    )


def create_style_matching_rubric(
    judge_client,
    judge_model: str,
    style_max: int,
    penalty_max: int,
    seed: int,
    author_id_bonus: int,
) -> vf.Rubric:
    formatted_prompt = style_adherence_prompt.format(
        style_max=style_max, penalty_max=penalty_max
    )

    judge_rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=formatted_prompt,
        # No response_format: rely on the prompt + tolerant parser + retry so this
        # works against any OpenAI-compatible model, including Tinfoil-hosted ones.
        judge_sampling_args={
            "temperature": 0.25,
            "max_tokens": 1500,
            "seed": seed,
            # gpt-oss is a reasoning model: keep reasoning minimal so the JSON
            # answer fits in the token budget. Reasoning lands in a separate
            # `reasoning_content` field, so `content` stays clean JSON.
            "extra_body": {"reasoning_effort": "low"},
        },
    )

    original_judge = judge_rubric.judge

    async def randomized_judge(prompt, completion, answer, state: State):
        if "rng" not in state:
            state["rng_seed"] = seed
            state["rng"] = random.Random(seed)

        order_swapped = state["rng"].choice([True, False])
        state["order_swapped"] = order_swapped

        if order_swapped:
            answer, completion = completion, answer

        return await original_judge(prompt, completion, answer, state)

    async def style_matching_reward(
        completion, state: State, info: Info, prompt=None, answer=None, **kwargs
    ) -> float:
        judge_retries = 0
        while True:
            try:
                state["judge_retries"] = judge_retries
                judge_retries += 1
                judge_response = await randomized_judge(
                    prompt=prompt, completion=completion, answer=answer, state=state
                )
                scores = validate_and_parse_style_adherence_judge_response(
                    judge_response,
                    style_max=style_max,
                    penalty_max=penalty_max,
                    order_swapped=state["order_swapped"],
                )
                state["judge_response"] = judge_response
                state["judge_scores"] = scores
                break
            except ValueError:
                if judge_retries > 5:
                    raise
                continue

        style_match = scores["style_match"]
        format_penalty = scores["format_penalty"]
        a_correct = scores["a_correct"]
        b_correct = scores["b_correct"]

        author_id_reward = 0
        if a_correct:
            author_id_reward += author_id_bonus / 2
        if b_correct:
            author_id_reward += author_id_bonus / 2

        total_reward = style_match + author_id_reward - abs(format_penalty)
        max_possible_reward = style_max + author_id_bonus
        normalized_reward = float(total_reward) / max_possible_reward
        normalized_reward = max(0.0, min(1.0, normalized_reward))

        metrics = {
            "style_match": style_match / style_max,
            "format_penalty": abs(format_penalty) / abs(penalty_max),
            "author_id_reward_normalized": author_id_reward / author_id_bonus,
            "a_correct": a_correct,
            "b_correct": b_correct,
            "match_pros": scores["match_pros"],
            "match_cons": scores["match_cons"],
        }
        state.update(metrics)
        info.update(metrics)
        return normalized_reward

    return vf.Rubric(funcs=[style_matching_reward], weights=[1.0])


def default_format_data_fn(example):
    prompt = f"""Continue this text in the same writing style.

{example['story_snippet']}"""
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "answer": example["actual_continuation"],
        "info": {
            "story_snippet": example["story_snippet"],
            "actual_continuation": example["actual_continuation"],
        },
        "task": "style-matching",
    }


def _default_judge_client():
    """Build a verified Tinfoil async client (the privacy-preserving judge)."""
    from tinfoil import AsyncTinfoilAI

    return AsyncTinfoilAI(
        api_key=os.environ.get("TINFOIL_API_KEY", "tinfoil"),
    ).client


def load_corpus_file(path: str) -> list[str]:
    """Read a user's writing from disk into a list of documents.

    Supports `.jsonl` (one JSON object per line with a "text" field), `.json`
    (a list of strings), and plain `.txt` (split into docs on blank lines).
    This is what prime-rl passes via `args = { corpus_path = ... }`.
    """
    text = open(path, encoding="utf-8").read()
    if path.endswith(".jsonl"):
        return [json.loads(line)["text"] for line in text.splitlines() if line.strip()]
    if path.endswith(".json"):
        return json.loads(text)
    return [doc.strip() for doc in text.split("\n\n") if doc.strip()]


def load_environment(
    dataset: Dataset | None = None,
    user_texts: list[str] | None = None,
    corpus_path: str | None = None,
    judge_client=None,
    judge_model: str | None = None,
    system_prompt: str | None = None,
    format_data_fn: Callable = default_format_data_fn,
    style_max: int = 10,
    penalty_max: int = -5,
    author_id_bonus: int = 2,
    seed: int = 42,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Load the personal-style environment.

    Args:
        dataset: explicit (story_snippet, actual_continuation) dataset.
        user_texts: a user's raw documents; chunked into a dataset if `dataset`
            is None.
        corpus_path: path to a user's writing on disk (jsonl/json/txt); read and
            chunked if `dataset` and `user_texts` are both None. This is the arg
            prime-rl forwards per user. If all three are None, a built-in sample.
        judge_client: OpenAI-compatible async client for the judge; defaults to a
            verified Tinfoil client.
        judge_model: judge model name; defaults to gpt-oss-120b.
    """
    judge_client = judge_client or _default_judge_client()
    judge_model = judge_model or "gpt-oss-120b"

    if dataset is None:
        if user_texts is None and corpus_path is not None:
            user_texts = load_corpus_file(corpus_path)
        dataset = (
            chunk_user_corpus(user_texts) if user_texts else create_sample_dataset()
        )
    dataset = dataset.map(format_data_fn)

    rubric = create_style_matching_rubric(
        judge_client=judge_client,
        judge_model=judge_model,
        style_max=style_max,
        penalty_max=penalty_max,
        author_id_bonus=author_id_bonus,
        seed=seed,
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
