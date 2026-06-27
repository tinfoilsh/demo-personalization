#!/usr/bin/env bash
# Hit a running control server end-to-end (no GPU, mock mode).
# Prereqs: the three services running (see deploy/README.md) + TINFOIL_API_KEY set.
#
#   PERSONALIZATION_DEMO_KEY=demo ENC_KEY=my-secret-key ./scripts/smoke_live.sh
set -euo pipefail

CONTROL="${CONTROL:-http://localhost:9000}"
PERSONALIZATION_DEMO_KEY="${PERSONALIZATION_DEMO_KEY:-demo}"
ENC_KEY="${ENC_KEY:-my-secret-key}"
H=(-H "x-demo-key: ${PERSONALIZATION_DEMO_KEY}" -H "content-type: application/json")

echo "== /train =="
curl -fsS "${H[@]}" \
  -d "{\"documents\":[\"I always start my mornings with a strong opinion and a weak coffee. The world, I find, rewards conviction over caffeine.\",\"Honestly? Most meetings could be a sentence. Two if you are feeling generous.\"],\"encryption_key\":\"${ENC_KEY}\"}" \
  "${CONTROL}/train"; echo

echo "== /status (poll until ready/failed) =="
for _ in $(seq 1 60); do
  s=$(curl -fsS "${H[@]}" -d "{\"encryption_key\":\"${ENC_KEY}\"}" "${CONTROL}/status"); echo "$s"
  echo "$s" | grep -qE '"status": ?"(ready|failed)"' && break
  sleep 5
done

echo "== /v1/chat/completions =="
curl -fsS "${H[@]}" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write me a one-line note about Mondays.\"}],\"encryption_key\":\"${ENC_KEY}\"}" \
  "${CONTROL}/v1/chat/completions"; echo
