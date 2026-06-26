#!/usr/bin/env bash
# Hit a running control server end-to-end (no GPU, mock mode).
# Prereqs: the three services running (see deploy/README.md) + TINFOIL_API_KEY set.
#
#   DEMO_KEY=demo ENC_KEY=my-secret-key ./scripts/smoke_live.sh
set -euo pipefail

CONTROL="${CONTROL:-http://localhost:9000}"
DEMO_KEY="${DEMO_KEY:-demo}"
ENC_KEY="${ENC_KEY:-my-secret-key}"
H=(-H "x-demo-key: ${DEMO_KEY}" -H "x-encryption-key: ${ENC_KEY}")

echo "== /train =="
curl -fsS "${H[@]}" -H "content-type: application/json" \
  -d '{"documents":["I always start my mornings with a strong opinion and a weak coffee. The world, I find, rewards conviction over caffeine.","Honestly? Most meetings could be a sentence. Two if you are feeling generous."]}' \
  "${CONTROL}/train"; echo

echo "== /status (poll until ready/failed) =="
for _ in $(seq 1 60); do
  s=$(curl -fsS "${H[@]}" "${CONTROL}/status"); echo "$s"
  echo "$s" | grep -qE '"status": ?"(ready|failed)"' && break
  sleep 5
done

echo "== /v1/chat/completions =="
curl -fsS "${H[@]}" -H "content-type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write me a one-line note about Mondays."}]}' \
  "${CONTROL}/v1/chat/completions"; echo
