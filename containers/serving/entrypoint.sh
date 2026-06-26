#!/bin/sh
# vLLM and huggingface_hub only read HF_TOKEN; prod provisions the gated-Llama
# token under its own name (LLAMA_HF_TOKEN), so bridge it here.
[ -n "$LLAMA_HF_TOKEN" ] && export HF_TOKEN="$LLAMA_HF_TOKEN"
exec vllm serve "$@"
