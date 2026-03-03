#!/bin/bash
# Verification script for the dev Loki server.
# Sends a test log entry mimicking usage_lib._send_to_loki() format,
# then queries it back to confirm ingestion.
#
# Usage:
#   bash sky/usage/dev/test-push.sh http://<IP>:9090
#   bash sky/usage/dev/test-push.sh http://<IP>:9090 http://<IP>:3225

set -euo pipefail

PUSH_URL="${1:?Usage: $0 <push-url> [viewer-url]}"
VIEWER_URL="${2:-}"
TIMESTAMP=$(date +%s)000000000  # nanoseconds

echo "=== Testing Loki Dev Server ==="
echo ""

# 1. Push a test log entry (same format as usage_lib._send_to_loki)
echo "1. Pushing test log entry to ${PUSH_URL}..."
PAYLOAD=$(cat <<EOF
{
  "streams": [{
    "stream": {
      "type": "usage",
      "environment": "dev",
      "test": "true"
    },
    "values": [
      ["${TIMESTAMP}", "{\"msg\": \"test-push from dev verification script\", \"ts\": ${TIMESTAMP}}"]
    ]
  }]
}
EOF
)

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "${PUSH_URL}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}")

if [ "${HTTP_CODE}" = "204" ]; then
  echo "   SUCCESS: Got HTTP 204 (expected)"
else
  echo "   FAIL: Got HTTP ${HTTP_CODE} (expected 204)"
  exit 1
fi

# 2. Query the data back if viewer URL is provided
if [ -n "${VIEWER_URL}" ]; then
  echo ""
  echo "2. Querying data back from ${VIEWER_URL}..."
  echo "   (waiting 3s for ingestion...)"
  sleep 3

  QUERY_RESULT=$(curl -s -G -u admin:skypilot-dev \
    "${VIEWER_URL}/loki/api/v1/query_range" \
    --data-urlencode 'query={type="usage", environment="dev", test="true"}' \
    --data-urlencode "start=${TIMESTAMP}" \
    --data-urlencode "limit=5")

  RESULT_STATUS=$(echo "${QUERY_RESULT}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "parse-error")

  if [ "${RESULT_STATUS}" = "success" ]; then
    echo "   SUCCESS: Query returned status=success"
    echo "   Response: ${QUERY_RESULT}" | head -c 500
    echo ""
  else
    echo "   WARN: Query returned status=${RESULT_STATUS}"
    echo "   Response: ${QUERY_RESULT}" | head -c 500
    echo ""
  fi
else
  echo ""
  echo "2. Skipping read verification (no viewer URL provided)."
  echo "   Re-run with: $0 ${PUSH_URL} http://<IP>:3225"
fi

echo ""
echo "=== Done ==="
