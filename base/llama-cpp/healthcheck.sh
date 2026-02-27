#!/bin/sh
# Healthcheck for llama-server
# Returns 0 if the server is responding, 1 otherwise

PORT="${FOUNDRY_PORT:-8080}"
curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1
