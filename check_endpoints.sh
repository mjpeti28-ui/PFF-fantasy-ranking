#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}

# Optional API key header
if [[ -n "${PFF_API_KEY:-}" ]]; then
  AUTH_HEADER=(-H "X-API-Key: ${PFF_API_KEY}")
else
  AUTH_HEADER=()
fi

# Helper to call curl, passing optional auth header only when present
curl_json() {
  if [[ ${#AUTH_HEADER[@]} -gt 0 ]]; then
    curl -sS "${AUTH_HEADER[@]}" "$@"
  else
    curl -sS "$@"
  fi
}

echo "== /players (top 5 RB by projected points) =="
PLAYER_URL="${BASE_URL}/players?pos=RB&metric=proj&limit=5"
curl_json "$PLAYER_URL" | jq '.'

echo "== /rankings (top 10 overall by rank) =="
RANKINGS_URL="${BASE_URL}/rankings?metric=rank&limit=10"
curl_json "$RANKINGS_URL" | jq '.'

echo "== /evaluate (include details, benchLimit=3) =="
EVAL_BODY='{"includeDetails": true, "benchLimit": 3}'
curl_json -X POST "${BASE_URL}/evaluate" \
  -H "Content-Type: application/json" \
  -d "$EVAL_BODY" \
  | jq '{evaluatedAt, playerCount, teams: .teams[:3], details: (.details // [])[:1]}'

echo "== /trade/evaluate sample =="
cat <<'JSON' > /tmp/trade-payload.json
{
  "teamA": "JohnBurrows_School",
  "teamB": "KWood_Super_Marios",
  "sendA": [
    {"group": "WR", "name": "George Pickens"}
  ],
  "sendB": [
    {"group": "WR", "name": "Drake London"}
  ],
  "includeDetails": true,
  "benchLimit": 2
}
JSON

curl_json -X POST "${BASE_URL}/trade/evaluate" \
  -H "Content-Type: application/json" \
  -d @/tmp/trade-payload.json \
  | jq '{evaluatedAt, teams, details}'

echo "== /stats/passing (top fantasyPts) =="
curl_json "${BASE_URL}/stats/passing?limit=5&sort=-fantasyPts" | jq '.'

echo "== /sources/projections (sample) =="
curl_json "${BASE_URL}/sources/projections?limit=5" | jq '.'
