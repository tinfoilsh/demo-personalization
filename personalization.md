# Personalization layer: "write like me" on confidential compute

Companion to [`core_server.md`](./core_server.md). That doc covers the infra (prime-rl → LoRA → multi-tenant vLLM serving). This doc covers the **demo itself**: what we train, the environment that defines "writes like you," what the single H200 can actually support, and how people reach it.

**North star:** a user uploads their own writing inside the enclave, we RL-post-train a per-user LoRA so the model writes like them, and we serve that adapter back. The point of doing it on confidential compute is that _the private writing never leaves the enclave — not even to grade itself._

---

## 1. Model — decided

- **Policy / base: `Qwen3-4B-Instruct-2507`**, LoRA. It's the prime-rl `alphabet_sort` baseline (proven: LoRA RL on a single H100 in ~1 hr), fluent enough at 4B for a convincing style demo, and small enough to leave the H200 with room for the trainer + a serving engine + many resident adapters.
- **Judge / reward model: a separate Tinfoil OpenAI-compatible endpoint** (another box, not the H200). Keeps the training H200 to trainer + policy-inference only, and keeps the user's text in-enclave (see §3).
- Fallback-up if 4B prose feels flat once we have a GPU: Qwen3-8B-Instruct-2507 or Llama-3.1-8B-Instruct (both fit on the H200 with LoRA+GRPO; Llama reads most "human"). Don't go ≥14B for colocated train+serve.
- LoRA rank: start 16–32 for style (rank↔reward isn't linear; 32 is a safe ceiling).

---

## 2. The environment

### 2a. What already exists: `apaz/writing-style-matching`

**Provenance: community, not official.** The `apaz/` namespace is an individual contributor's — this is _not_ a Prime Intellect environment (those are `primeintellect/*`) and it didn't make Prime's curated community list ([`PrimeIntellect-ai/prime-environments`](https://github.com/PrimeIntellect-ai/prime-environments)). It's one person's upload, served from their own Hub index, README still a scaffold placeholder. Doesn't matter for us — the logic is sound and we fork it anyway (§2b). Curated alternatives to crib judge-prompt ideas from: `creative_writing`, `llm_writer_negative_style`, `llm_writing_detection`, `writing_bench`.

The closest existing Hub environment, and an almost-exact template. Mechanics (read from source via `prime env inspect apaz/writing-style-matching`):

- **Dataset**: rows of `(story_snippet, actual_continuation)` — the continuation is the _real author's_ text.
- **Task**: "Continue this story in the same writing style."
- **Reward**: an LLM judge sees two continuations — A = the real author's (`answer`), B = the model's (`response`), order randomized — and returns:
  1. `style_match` 0–10 (vocab, sentence rhythm, tone, "same author?")
  2. `format_penalty` 0 to −5 (penalizes "Here's the continuation:" meta-junk, quoting the prompt back, repetition)
  3. **author identification** — judge guesses whether each continuation shares the snippet's author. Bonus reward for fooling the judge into thinking the model's output is the same author. This is a built-in discriminator, for free, inside one judge call.
- Final reward = `(style_match + author_id_bonus − |format_penalty|)` normalized to [0,1].
- Judge defaults to `claude-haiku-4-5` **hardcoded to `api.anthropic.com`** (see §3).

### 2b. What we build: a thin fork

Personalization is almost entirely a **data-pipeline** change — the RL/reward logic already does the right thing. Two small code changes on top:

1. **Swap the dataset for the user's own writing.** Chunk their corpus into `(prefix, real_continuation)` pairs; pass as `dataset=`. The model learns to continue _the user's_ text indistinguishably from the real thing.
2. **Fork the single env file** (`writing_style_matching.py`, ~15 KB) to (a) make the judge endpoint configurable → Tinfoil, and (b) generalize wording from "story" → "text" so it fits emails/messages/essays, not just fiction.

Sketch:

```python
# personal_style_env.py  (fork of writing_style_matching.py)
import os
from datasets import Dataset
from openai import AsyncOpenAI

def chunk_user_corpus(texts: list[str], min_words=80, max_pairs=200) -> Dataset:
    """Split a user's writing into (prefix, continuation) pairs."""
    rows = []
    for t in texts:
        w = t.split()
        if len(w) < min_words:
            continue
        cut = len(w) // 2
        rows.append((" ".join(w[:cut]), " ".join(w[cut:])))
    rows = rows[:max_pairs]
    return Dataset.from_dict({
        "story_snippet":       [p for p, _ in rows],   # = prefix
        "actual_continuation": [c for _, c in rows],   # = real user text
    })

# In the forked load_environment, replace the hardcoded Anthropic client with:
judge_client = AsyncOpenAI(
    base_url=os.environ["TINFOIL_JUDGE_BASE_URL"],   # in-enclave endpoint
    api_key=os.environ["TINFOIL_JUDGE_API_KEY"],
)
# judge_model = os.environ["TINFOIL_JUDGE_MODEL"]
# and reword `style_adherence_prompt` / default prompt: "story" -> "text"
```

That's the whole environment. Everything else (rubric, randomized A/B order, author-ID discriminator, format penalty, JSON-schema judge parsing with retries) we keep as-is.

**Data needed:** for signal, ~50–200 prefix/continuation pairs; the concept demo survives on less. Held out a slice of the user's text for eval (§4).

---

## 3. Privacy — the whole point

The stock env ships every `story_snippet`, the real continuation, and every model output to **Anthropic's API** to be graded. On confidential compute that leaks exactly the private data the enclave exists to protect.

**Fix:** point the judge at a Tinfoil-hosted OpenAI-compatible model. The base*url is hardcoded in the stock file, so this requires the fork in §2b (not just an env var). With that, the user's writing + the policy's outputs stay in-enclave end to end. This \_is* the demo's privacy narrative: "your writing never leaves confidential compute, not even to score itself."

---

## 4. What the single H200 supports: train + serve at once

Goal: train one adapter while serving ≥3 already-trained adapters, ideally simultaneously. **Feasible for a demo.**

- **Memory (141 GB) is not the bottleneck.** Two vLLM engines (rollout + serving) at ~8 GB base each + trainer (~15–25 GB) + adapters (tens of MB each) leaves large headroom.
- **Compute contention is the only real cost.** SMs are shared, so a training compute burst can add latency jitter to a concurrent serving request. prime-rl already time-shares internally (rollout ↔ train-step ↔ weight-broadcast alternate), and demo serving traffic is light — so they coexist; expect occasional jitter, not failure.

**Realistic topology** (a prime-rl run trains _one_ policy at a time):

```
H200 (141 GB)
├── Serving vLLM        : Qwen3-4B base + N user adapters (multi-LoRA, runtime hot-load on)  ← demo traffic
└── prime-rl run        : trainer (FSDP2 LoRA) + rollout vLLM                                 ← trains the NEXT user's adapter
        └── judge calls ──→ separate Tinfoil endpoint (off-box)
```

When a training run finishes, **hot-load** the new adapter into the serving engine (`VLLM_ALLOW_RUNTIME_LORA_UPDATING`, per `core_server.md` §7) — no restart. This is the Trajectory pattern (sync RL Qwen3-4B on one H200, multiple LoRAs hot in memory).

**Fallback if jitter ever bothers a live audience:** alternate — train, then flip the card to pure serving. Simplest, zero contention, but not simultaneous.

---

## 5. Access layer (deferred — "figure out as we go")

- Expose a plain **OpenAI-compatible API** (vLLM multi-LoRA gives this for free; per-user adapter = the `model` field). This is the primary interface.
- If live demos get awkward, a thin frontend that just calls that API. No HTML/login/upload portal needed for the concept.

---

## 6. Open items (revisit when the GPU lands)

- Fork `writing_style_matching.py`; wire judge to Tinfoil endpoint; reword "story" → "text".
- Build `chunk_user_corpus` + pick a real default corpus for the first demo user.
- Empirically tune: `--inference.gpu-memory-utilization` split between rollout and serving engines, LoRA rank, `max-loras`/`max-cpu-loras`.
- Confirm whether one vLLM can sanely double as rollout + serving (saves a base copy) or whether two engines is worth the isolation. Default to two.
- Decide judge model on Tinfoil (capable instruct model; the stock env assumes Haiku-class quality).
