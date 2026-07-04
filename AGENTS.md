# AI Agent Integration Guide

This guide covers how to connect AI agents, coding assistants, and multi-agent frameworks to a Foundry inference server. Foundry exposes a standard OpenAI-compatible API at `http://localhost:8080/v1`.

## Table of Contents

- [API Reference](#api-reference)
- [Coding Agents](#coding-agents)
- [Multi-Agent Frameworks](#multi-agent-frameworks)
- [Chat Interfaces](#chat-interfaces)
- [Direct API Usage](#direct-api-usage)
- [Choosing a Model](#choosing-a-model)
- [Performance Considerations](#performance-considerations)
- [Structured Output](#structured-output)
- [Tool Calling / Function Calling](#tool-calling--function-calling)
- [Thinking / Reasoning Mode](#thinking--reasoning-mode)
- [Streaming](#streaming)
- [Multi-GPU Agent Routing](#multi-gpu-agent-routing)
- [Troubleshooting](#troubleshooting)

## API Reference

Foundry serves the full [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/completions):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

Base URL: `http://localhost:8080` (configurable via `FOUNDRY_PORT`)

No API key is required by default. If your client demands one, any non-empty string works (e.g. `sk-local`).

## Coding Agents

### OpenCode

[OpenCode](https://opencode.ai) connects via the `@ai-sdk/openai-compatible` provider.

> **Important:** Use `@ai-sdk/openai-compatible`, not `@ai-sdk/openai`. The latter crashes on
> models that emit `<think>` tokens (see [Troubleshooting](TROUBLESHOOTING.md#opencode-text-part-msg-not-found)).

```json
// opencode.json (project root or ~/.config/opencode/opencode.json)
{
  "$schema": "https://opencode.ai/config.json",
  "model": "foundry/qwen3.6-35b-a3b-nvfp4",
  "provider": {
    "foundry": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Foundry",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "sk-local"
      },
      "models": {
        "qwen3.6-35b-a3b-nvfp4": {
          "name": "Qwen 3.5 9B",
          "limit": {
            "context": 229376,
            "output": 32768
          }
        }
      }
    }
  }
}
```

The model ID must match what `/v1/models` returns (check with `curl http://localhost:8080/v1/models`).

### Cursor

Settings > Models > OpenAI API Base:

```
Base URL: http://localhost:8080/v1
API Key:  sk-local
Model:    qwen3.5-9b
```

Cursor uses streaming by default. Foundry supports SSE streaming natively. vLLM's continuous batching runs Cursor's background indexing and active chat simultaneously without blocking.

### Continue (VS Code / JetBrains)

```json
// ~/.continue/config.json
{
  "models": [
    {
      "title": "Foundry Qwen",
      "provider": "openai",
      "model": "qwen3.5-9b",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "sk-local"
    }
  ]
}
```

### Aider

```bash
aider --openai-api-base http://localhost:8080/v1 \
      --openai-api-key sk-local \
      --model openai/qwen3.5-9b
```

Or set environment variables:

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=sk-local
aider --model openai/qwen3.5-9b
```

### Cline (VS Code)

Settings > Cline > API Provider: OpenAI Compatible

```
Base URL: http://localhost:8080/v1
API Key:  sk-local
Model ID: qwen3.5-9b
```

## Multi-Agent Frameworks

vLLM's continuous batching makes Foundry particularly suited for multi-agent workflows where multiple agents share one model: up to 8 concurrent sequences at ~1,228 tok/s aggregate on RTX 5090.

### CrewAI

```python
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"
os.environ["OPENAI_API_KEY"] = "sk-local"
os.environ["OPENAI_MODEL_NAME"] = "qwen3.5-9b"

from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find and analyze information",
    backstory="Expert at finding and synthesizing information.",
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear technical documentation",
    backstory="Expert at turning research into documentation.",
    verbose=True,
)

research_task = Task(
    description="Research the topic: {topic}",
    expected_output="Detailed research notes",
    agent=researcher,
)

writing_task = Task(
    description="Write documentation based on the research",
    expected_output="A well-structured technical document",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "GPU inference optimization"})
```

CrewAI can run 4 agents simultaneously at ~307 tok/s each (RTX 5090, MTP x4 speculative decoding).

### AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent

config_list = [
    {
        "model": "qwen3.5-9b",
        "base_url": "http://localhost:8080/v1",
        "api_key": "sk-local",
    }
]

assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"},
)

user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers efficiently.",
)
```

### LangChain / LangGraph

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-local",
    model="qwen3.5-9b",
    streaming=True,
)

response = llm.invoke("Explain quantum computing in simple terms.")
print(response.content)
```

### Smolagents (Hugging Face)

```python
from smolagents import ToolCallingAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id="qwen3.5-9b",
    api_base="http://localhost:8080/v1",
    api_key="sk-local",
)

agent = ToolCallingAgent(tools=[], model=model)
agent.run("What is the capital of France?")
```

## Chat Interfaces

### Open WebUI

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e OPENAI_API_KEY=sk-local \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

Open WebUI supports multi-user chat with conversation history. Concurrent user sessions are batched by vLLM automatically.

### text-generation-webui (oobabooga)

Settings > Model > OpenAI:

```
API URL:  http://localhost:8080
API Key:  sk-local
```

### SillyTavern

Settings > API > Chat Completion (OpenAI):

```
API URL:     http://localhost:8080
API Key:     sk-local
Model:       qwen3.5-9b
```

## Direct API Usage

### Python (openai library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-local",
)

response = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)

print(response.choices[0].message.content)
```

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

### TypeScript / Node.js

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "sk-local",
});

const response = await client.chat.completions.create({
  model: "qwen3.5-9b",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response.choices[0].message.content);
```

## Choosing a Model

| Use case | Recommended model | Why |
|----------|-------------------|-----|
Foundry ships one model: **qwen3.6-35b-a3b-nvfp4** (MoE, ~3B active). It covers coding agents, multi-agent orchestration, tool calling, and long-context work (224K) in a single deployment; thinking mode (`reasoning_content`) is available per request for reasoning-heavy tasks.

## Performance Considerations

### Latency budget

Single-stream decode latency (time to generate one token):

| Model | Latency per token | Tokens per second |
|-------|-------------------|-------------------|
| qwen3.6-35b-a3b-nvfp4 | ~2.6 ms | ~384 tok/s (MTP x4) |

At ~384 tok/s single-stream the typing experience is instant for interactive agents; concurrent agent fleets aggregate to ~1,228 tok/s.

### Prompt processing

Prefill processes a ~1K-token prompt in ~0.11 s on RTX 5090. Keep system prompts concise to minimize time-to-first-token.

### Concurrent agent scaling

qwen3.6-35b-a3b-nvfp4 (RTX 5090, MTP x4):
- 1 agent:  ~384 tok/s
- 4 agents: ~1,228 tok/s aggregate (~307 tok/s each)

vLLM batches up to 8 concurrent sequences (`--max-num-seqs`); beyond that, requests queue. Consider multi-GPU routing (below) for higher concurrency.

### Context window usage

VRAM scales with context usage. The default RTX 5090 profiles are tuned for maximum context:

| Model | Default context | VRAM at idle | VRAM at full context |
|-------|----------------|--------------|---------------------|
| qwen3.6-35b-a3b-nvfp4 | 224K | 22 GB | ~29.0 GB |

To reduce VRAM usage, lower the context window:

```bash
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  -e FOUNDRY_CTX_LENGTH=32768 \
  ghcr.io/infernet-org/foundry/qwen3.5-9b:latest
```

## Structured Output

All three models support JSON mode for structured outputs:

```python
response = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{
        "role": "user",
        "content": "List 3 programming languages with their year of creation. Respond in JSON."
    }],
    response_format={"type": "json_object"},
)
```

For grammar-constrained generation (guaranteed schema compliance), use the `grammar` parameter:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Generate a person record"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
          },
          "required": ["name", "age"]
        }
      }
    }
  }'
```

## Tool Calling / Function Calling

Qwen3.6 supports tool calling via its chat template (`--jinja`-style templating is built into vLLM serving).

```python
response = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }],
)

# The model will respond with a tool_call if it decides to use the function
tool_calls = response.choices[0].message.tool_calls
if tool_calls:
    print(f"Function: {tool_calls[0].function.name}")
    print(f"Args: {tool_calls[0].function.arguments}")
```

All models support Jinja chat templates for tool calling. The entrypoint enables `--jinja` by default.

## Thinking / Reasoning Mode

qwen3.6-35b-a3b-nvfp4 supports a thinking mode: reasoning is returned separately in the `reasoning_content` field (qwen3 reasoning parser).

The server returns thinking content in the `reasoning_content` field:

```python
response = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "What is 127 * 389?"}],
    max_tokens=512,
)

msg = response.choices[0].message
if hasattr(msg, "reasoning_content") and msg.reasoning_content:
    print(f"Thinking: {msg.reasoning_content}")
print(f"Answer: {msg.content}")
```

## Streaming

All endpoints support Server-Sent Events (SSE) streaming for real-time token delivery:

```python
stream = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "Write a poem about GPUs."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

The host tuning script (`sudo ./scripts/host-setup.sh`) configures BBR congestion control and busy polling for minimal streaming latency. Time-to-first-token is typically ~50-200 ms depending on prompt length.

## Multi-GPU Agent Routing

For workloads requiring more concurrency than one GPU provides, run multiple Foundry instances and load-balance across them.

### Simple round-robin with nginx

```nginx
upstream foundry {
    server localhost:8080;  # GPU 0
    server localhost:8081;  # GPU 1
}

server {
    listen 80;
    location /v1/ {
        proxy_pass http://foundry;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;  # Required for SSE streaming
    }
}
```

### Agent-per-GPU isolation

For deterministic routing (each agent always hits the same GPU):

```python
import os

# Route based on agent ID (match --max-num-seqs, default 8)
seqs_per_gpu = 8
gpu_endpoints = [
    "http://localhost:8080/v1",  # GPU 0
    "http://localhost:8081/v1",  # GPU 1
]

def get_client(agent_id: int) -> OpenAI:
    endpoint = gpu_endpoints[agent_id // seqs_per_gpu]
    return OpenAI(base_url=endpoint, api_key="sk-local")
```

## Troubleshooting

For a comprehensive troubleshooting guide, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Quick reference for the most common issues:

### Model loads on CPU instead of GPU

If you see `no devices with dedicated memory found` in the logs, the CUDA backend failed to load. Check:

1. **NVIDIA driver**: `nvidia-smi` should work on the host
2. **Container GPU access**: `docker run --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi`
3. **CUDA libraries**: The image must contain `libcudart.so.12`, `libcublas.so.12`, and `libcublasLt.so.12` in `/app/`

### Server responds slowly or hangs

1. Check if all layers are on GPU: look for `offloaded N/N layers to GPU` in container logs
2. Check VRAM: `nvidia-smi` -- if VRAM is full, reduce context with `FOUNDRY_CTX_LENGTH`
3. Check queue depth: `curl http://localhost:8080/metrics | grep vllm:num_requests`

### OpenCode: "text part msg_... not found"

Use `@ai-sdk/openai-compatible` (not `@ai-sdk/openai`) and ensure the server has `--reasoning-format none`. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#opencode-text-part-msg-not-found) for details.

### Connection refused

1. Container might still be loading the model. Check `docker logs <container>` for progress.
2. First run downloads the model (6-22 GB depending on model) which can take several minutes.
3. Port conflict: use `-p 8081:8080` to map to a different host port.

### Out of VRAM

Reduce context window or switch to a smaller quantization:

```bash
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  -e FOUNDRY_CTX_LENGTH=16384 \
  ghcr.io/infernet-org/foundry/qwen3.5-9b:latest
```

This model requires an NVFP4-capable GPU (Blackwell RTX 50xx or Hopper) with 32 GB+ VRAM; there is no smaller variant in this repo.
