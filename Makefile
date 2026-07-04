# ==============================================================================
# Foundry Makefile
# ==============================================================================

REGISTRY ?= ghcr.io/infernet-org/foundry
# The only supported model (vLLM + NVFP4, Blackwell/Hopper GPUs)
MODEL ?= qwen3.6-35b-a3b-nvfp4
MODEL_TAG ?= $(REGISTRY)/$(MODEL)
PORT ?= 8080
MODELS_DIR ?= $(HOME)/.cache/foundry

# Shared docker-run flags (keep in sync with docker-compose.yml)
DOCKER_RUN_FLAGS = --gpus all \
	--shm-size 8g \
	--sysctl net.core.somaxconn=4096 \
	--sysctl net.ipv4.tcp_keepalive_time=60 \
	-p $(PORT):8080 \
	-v $(MODELS_DIR):/models

.PHONY: help build run run-profile test benchmark monitoring down push clean clean-models download

help: ## Show this help
	@echo "Model: qwen3.6-35b-a3b-nvfp4 (vLLM + NVFP4, requires Blackwell or Hopper GPU)"
	@echo "Usage: make run"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Build -------------------------------------------------------------------

build: ## Build the model image
	docker build \
		-t $(MODEL_TAG):latest \
		models/$(MODEL)/

# --- Run ---------------------------------------------------------------------

up: ## Start via docker compose (detatched)
	FOUNDRY_MODEL=$(MODEL) docker compose up -d

monitoring: ## Start via docker compose with full monitoring stack
	FOUNDRY_MODEL=$(MODEL) docker compose --profile monitoring up -d

down: ## Stop all docker compose services
	docker compose --profile monitoring down

run: ## Run the model container directly (auto-detect GPU)
	@mkdir -p $(MODELS_DIR)
	docker run $(DOCKER_RUN_FLAGS) \
		--name foundry-$(MODEL) \
		--rm \
		$(MODEL_TAG):latest

run-profile: ## Run with explicit profile (PROFILE=rtx5090)
	@mkdir -p $(MODELS_DIR)
	docker run $(DOCKER_RUN_FLAGS) \
		-e FOUNDRY_PROFILE=$(PROFILE) \
		--name foundry-$(MODEL) \
		--rm \
		$(MODEL_TAG):latest

# --- Test --------------------------------------------------------------------

test: ## Smoke test: start container, wait for health, send one request
	@echo "Starting container..."
	@mkdir -p $(MODELS_DIR)
	@docker run -d $(DOCKER_RUN_FLAGS) \
		--name foundry-test-$(MODEL) \
		$(MODEL_TAG):latest
	@echo "Waiting for server to be ready..."
	@for i in $$(seq 1 300); do \
		if curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1; then \
			echo "Server ready after $$i seconds"; \
			break; \
		fi; \
		if [ $$i -eq 300 ]; then \
			echo "Timeout waiting for server"; \
			docker logs foundry-test-$(MODEL); \
			docker rm -f foundry-test-$(MODEL); \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@echo "Sending test request..."
	@curl -s http://localhost:$(PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"$(MODEL)","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}' \
		| python3 -m json.tool
	@echo ""
	@echo "Test passed. Cleaning up..."
	@docker rm -f foundry-test-$(MODEL)

# --- Download ----------------------------------------------------------------

download: ## Download the model weights (~22 GB snapshot)
	./scripts/download-model.sh

# --- Benchmark ---------------------------------------------------------------

benchmark: ## Run benchmark against a running server (PORT=8080)
	python3 scripts/benchmark.py --url http://localhost:$(PORT) --mode all

# --- Push --------------------------------------------------------------------

push: ## Push model image to GHCR
	docker push $(MODEL_TAG):latest

# --- Clean -------------------------------------------------------------------

clean: ## Remove local images
	-docker rmi $(MODEL_TAG):latest

clean-models: ## Remove downloaded models (incl. legacy GGUFs from pre-vLLM foundry)
	rm -rf "$(MODELS_DIR)"/Qwen3.6-35B-A3B-NVFP4 "$(MODELS_DIR)"/*.gguf
