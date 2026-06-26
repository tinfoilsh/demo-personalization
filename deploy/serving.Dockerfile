# Serving container: long-lived multi-LoRA vLLM, OpenAI-compatible. Holds the
# base model + every user's finished adapter. Runtime LoRA updating lets the
# control server hot-load new adapters without a restart.
FROM vllm/vllm-openai:latest

ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
ENV NVIDIA_VISIBLE_DEVICES=0

EXPOSE 8000
# The base image's entrypoint is the vLLM OpenAI server; these are its args.
# --gpu-memory-utilization is kept low so the trainer's rollout vLLM fits on the
# same H200 (see personalization.md §4). Adapters are mounted at runtime, so they
# are NOT baked into this (attested) image — they're private state on /state.
CMD ["--model", "Qwen/Qwen3-4B-Instruct-2507", \
     "--enable-lora", \
     "--max-lora-rank", "32", \
     "--max-loras", "4", \
     "--max-cpu-loras", "16", \
     "--gpu-memory-utilization", "0.4"]
