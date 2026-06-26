# personal_style

A `verifiers` RL environment that rewards a model for **writing like a specific user**.

The model is given the first half of a piece of the user's writing and must continue it.
An LLM judge then compares the model's continuation against the user's *real* continuation
on style match (vocabulary, rhythm, tone, voice) and tries to identify which continuation
shares the snippet's author — so the model is rewarded both for matching the style and for
fooling the judge into thinking its output is genuinely the user's.

## Attribution

Fork of the community environment **[`apaz/writing-style-matching`](https://app.primeintellect.ai/dashboard/environments/apaz/writing-style-matching)**.
The rubric design (judge-scored style match + author-identification discriminator) is apaz's.

Changes here:
- Judge runs on a **Tinfoil enclave** (verified, OpenAI-compatible) instead of Anthropic, so
  the user's writing never leaves confidential compute.
- Wording generalized from "story" → "text" (emails, messages, essays — not just fiction).
- `chunk_user_corpus()` builds a dataset from a user's own documents.
- Judge `response_format` dropped in favor of a tolerant parser + retry, so it runs against
  any OpenAI-compatible model.

## Usage

```python
from personal_style import load_environment

# Built-in sample data, judge defaults to a verified Tinfoil client:
env = load_environment()

# Or from a user's own writing:
env = load_environment(user_texts=["...my email...", "...my essay...", ...])
```

`load_environment` args: `dataset`, `user_texts`, `judge_client`, `judge_model`,
`system_prompt`, `style_max` (10), `penalty_max` (-5), `author_id_bonus` (2), `seed` (42).

Reward is normalized to [0, 1]: `(style_match + author_id_bonus − |format_penalty|) / (style_max + author_id_bonus)`.
