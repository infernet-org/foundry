# ==============================================================================
# Foundry Makefile
# ==============================================================================

REGISTRY ?= ghcr.io/infernet-org/foundry
MODEL ?= qwen3.5-35b-a3b
CUDA_VERSION ?= 12.8.0
CUDA_ARCH ?= 86;89;120
BASE_TAG ?= $(REGISTRY)/base-llama-cpp
MODEL_TAG ?= $(REGISTRY)/$(MODEL)
PORT ?= 8080
MODELS_DIR ?= $(HOME)/.cache/foundry

.PHONY: help build-base build run test push clean download

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Build -------------------------------------------------------------------

build-base: ## Build the base llama.cpp image
	docker build \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg CUDA_ARCH="$(CUDA_ARCH)" \
		-t $(BASE_TAG):latest \
		-t $(BASE_TAG):cu$(subst .,,$(word 1,$(subst ., ,$(CUDA_VERSION)))$(word 2,$(subst ., ,$(CUDA_VERSION)))) \
		base/llama-cpp/

build-base-cu124: ## Build base image for CUDA 12.4 (RTX 30/40 series)
	$(MAKE) build-base CUDA_VERSION=12.4.1 CUDA_ARCH="86;89"

build: build-base ## Build the model image
	docker build \
		--build-arg BASE_IMAGE=$(BASE_TAG):latest \
		-t $(MODEL_TAG):latest \
		models/$(MODEL)/

# --- Run ---------------------------------------------------------------------

run: ## Run the model container (auto-detect GPU)
	@mkdir -p $(MODELS_DIR)
	docker run --gpus all \
		--shm-size 2g \
		-p $(PORT):8080 \
		-v $(MODELS_DIR):/models \
		--name foundry-$(MODEL) \
		--rm \
		$(MODEL_TAG):latest

run-profile: ## Run with explicit profile (PROFILE=rtx4090)
	@mkdir -p $(MODELS_DIR)
	docker run --gpus all \
		--shm-size 2g \
		-p $(PORT):8080 \
		-v $(MODELS_DIR):/models \
		-e FOUNDRY_PROFILE=$(PROFILE) \
		--name foundry-$(MODEL) \
		--rm \
		$(MODEL_TAG):latest

# --- Test --------------------------------------------------------------------

test: ## Smoke test: start container, wait for health, send one request
	@echo "Starting container..."
	@mkdir -p $(MODELS_DIR)
	@docker run --gpus all -d \
		--shm-size 2g \
		-p $(PORT):8080 \
		-v $(MODELS_DIR):/models \
		--name foundry-test-$(MODEL) \
		$(MODEL_TAG):latest
	@echo "Waiting for server to be ready..."
	@for i in $$(seq 1 60); do \
		if curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1; then \
			echo "Server ready after $$i seconds"; \
			break; \
		fi; \
		if [ $$i -eq 60 ]; then \
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

download: ## Download the GGUF model file
	./scripts/download-model.sh

# --- Push --------------------------------------------------------------------

push: ## Push images to GHCR
	docker push $(BASE_TAG):latest
	docker push $(MODEL_TAG):latest

push-all: ## Push all tags to GHCR
	docker push --all-tags $(BASE_TAG)
	docker push --all-tags $(MODEL_TAG)

# --- Clean -------------------------------------------------------------------

clean: ## Remove local images
	-docker rmi $(MODEL_TAG):latest
	-docker rmi $(BASE_TAG):latest

clean-models: ## Remove downloaded models
	rm -rf $(MODELS_DIR)/*.gguf
