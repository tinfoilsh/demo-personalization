#!/usr/bin/env bash
# Build a GPU service image locally to catch Dockerfile/dependency errors fast,
# without the merge -> tag -> release cycle. The prime-rl base is amd64-only, so
# on Apple Silicon this builds under emulation (the one-time base pull is slow;
# overlay rebuilds after a code edit are seconds).
#
#   ./scripts/build_local.sh                 # trainer (default)
#   ./scripts/build_local.sh serving         # serving
#
# It only proves the image assembles + deps resolve — it can't run training
# (no GPU). For that, deploy to the GPU CVM.
set -euo pipefail
cd "$(dirname "$0")/.."

svc="${1:-trainer}"
docker buildx build \
  --platform linux/amd64 \
  -f "containers/${svc}/Dockerfile.gpu" \
  -t "dp/${svc}:local" \
  --load .
