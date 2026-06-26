"""Split a long text into ~uniform word-count chunks for the personal-style env.

The env (personal_style.chunk_user_corpus) turns each corpus doc into ONE
training example by cutting it in half: first half = prompt prefix, second half =
the real continuation the judge scores against. So docs want to be uniform and
sized so each half fits seq_len/max_completion_tokens — ~300 words is a good
middle (prefix ~150 / continuation ~150). We pack whole paragraphs up to the
target, never splitting mid-paragraph, and drop any final scrap under min_words.

    python chunk.py [input.txt] [output.jsonl] [--target 300] [--min 80]
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def chunk(text: str, target: int, min_words: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for para in paragraphs:
        buf.append(para)
        count += len(para.split())
        if count >= target:
            chunks.append("\n".join(buf))
            buf, count = [], 0
    if buf:  # flush remainder, merging into the last chunk if it's too short
        tail = "\n".join(buf)
        if len(tail.split()) >= min_words or not chunks:
            chunks.append(tail)
        else:
            chunks[-1] += "\n" + tail
    return [c for c in chunks if len(c.split()) >= min_words]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=str(HERE / "at_the_mountains_of_madness.txt"))
    ap.add_argument("output", nargs="?", default=str(HERE / "corpus.jsonl"))
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--min", type=int, default=80, dest="min_words")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    chunks = chunk(text, args.target, args.min_words)
    with open(args.output, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c}) + "\n")

    wc = [len(c.split()) for c in chunks]
    print(f"{len(chunks)} chunks -> {args.output}")
    print(f"words/chunk: min={min(wc)} median={sorted(wc)[len(wc)//2]} max={max(wc)}")


if __name__ == "__main__":
    main()
