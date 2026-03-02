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

[OpenCode](https://opencode.ai) connects directly via OpenAI-compatible providers.

```json
// ~/.config/opencode/config.json
{
  "provider": {
    "foundry": {
      "name": "Foundry",
      "type": "openai",
      "url": "http://localhost:8080/v1",
      "models": {
        "qwen": {
          "id": "qwen3.5-35b-a3b",
          "name": "Qwen 3.5 35B A3B"
        },
        "qwen-coder": {
          "id": "qwen3-coder-30b-a3b",
          "name": "Qwen 3 Coder 30B A3B"
        }
      }
    }
  }
}
```

### Cursor

Settings > Models > OpenAI API Base:

```
Base URL: http://localhost:8080/v1
API Key:  sk-local
Model:    qwen3.5-35b-a3b
```

Cursor uses streaming by default. Foundry supports SSE streaming natively. With multiple parallel slots, you can run Cursor's background indexing and active chat simultaneously without blocking.

### Continue (VS Code / JetBrains)

```json
// ~/.continue/config.json
{
  "models": [
    {
      "title": "Foundry Qwen",
      "provider": "openai",
      "model": "qwen3.5-35b-a3b",
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
      --model openai/qwen3.5-35b-a3b
```

Or set environment variables:

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=sk-local
aider --model openai/qwen3.5-35b-a3b
```

### Cline (VS Code)

Settings > Cline > API Provider: OpenAI Compatible

```
Base URL: http://localhost:8080/v1
API Key:  sk-local
Model ID: qwen3.5-35b-a3b
```

## Multi-Agent Frameworks

Foundry's parallel inference slots make it particularly suited for multi-agent workflows where multiple agents share a single model. Each slot processes requests independently with minimal throughput degradation.

### CrewAI

```python
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"
os.environ["OPENAI_API_KEY"] = "sk-local"
os.environ["OPENAI_MODEL_NAME"] = "qwen3.5-35b-a3b"

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

With 3 parallel slots, CrewAI can run 3 agents simultaneously at ~168 tok/s each (Qwen3-Coder MoE) or ~33 tok/s each with Hermes Dense (4 slots).

### AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent

config_list = [
    {
        "model": "qwen3.5-35b-a3b",
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
    model="qwen3.5-35b-a3b",
    streaming=True,
)

response = llm.invoke("Explain quantum computing in simple terms.")
print(response.content)
```

### Smolagents (Hugging Face)

```python
from smolagents import ToolCallingAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id="qwen3.5-35b-a3b",
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

Open WebUI supports multi-user chat with conversation history. Each user session uses one of Foundry's inference slots.

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
Model:       qwen3.5-35b-a3b
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
    model="qwen3.5-35b-a3b",
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
    "model": "qwen3.5-35b-a3b",
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
  model: "qwen3.5-35b-a3b",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response.choices[0].message.content);
```

## Choosing a Model

| Use case | Recommended model | Why |
|----------|-------------------|-----|
| **Coding agents** (OpenCode, Cursor, Aider) | Qwen3-Coder-30B-A3B | Fastest decode (275 tok/s), purpose-built for code, tool calling support |
| **Multi-agent orchestration** (CrewAI, AutoGen) | Qwen3-Coder-30B-A3B | 3-concurrent at 497 tok/s aggregate, best MoE batching efficiency |
| **General coding + long context** | Qwen3.5-35B-A3B | 192K effective context for large codebases, hybrid recurrent architecture |
| **Reasoning-heavy tasks** | Hermes-4.3-36B | Thinking mode with `<think>` tags, stronger reasoning on hard problems |
| **Tool use / function calling** | Qwen3-Coder-30B-A3B or Hermes-4.3-36B | Both have strong tool calling; Coder is 4x faster, Hermes more reliable on complex schemas |
| **Roleplay / creative writing** | Hermes-4.3-36B | NousResearch fine-tune optimized for personality and narrative |
| **Long document Q&A** | Qwen3.5-35B-A3B | 192K context window, recurrent layers handle long sequences efficiently |
| **16 GB VRAM GPUs** | Qwen3-Coder-30B-A3B | Smallest disk footprint (17.7 GB), MoE expert offloading works on 16 GB |

## Performance Considerations

### Latency budget

Single-stream decode latency (time to generate one token):

| Model | Latency per token | Tokens per second |
|-------|-------------------|-------------------|
| Qwen3-Coder-30B-A3B | ~3.6 ms | ~275 tok/s |
| Qwen3.5-35B-A3B | ~5.5 ms | ~181 tok/s |
| Hermes-4.3-36B | ~15.5 ms | ~64 tok/s |

For interactive coding agents, Qwen3-Coder delivers the fastest typing experience. Qwen3.5 trades some speed for 192K effective context. For batch/background tasks where latency is less critical, Hermes' stronger reasoning may be worth the tradeoff.

### Prompt processing

Prompt processing (prefill) runs at ~1,163 tok/s for Qwen on RTX 5090. A 10K token prompt takes ~8.6 seconds to process. Keep system prompts concise to minimize time-to-first-token.

### Concurrent agent scaling

Qwen3-Coder-30B-A3B (fastest, 3 slots):
```
1 agent:  275 tok/s  (100% per-agent speed)
2 agents: 405 tok/s  (~204 tok/s each, 74% per-agent)
3 agents: 497 tok/s  (~168 tok/s each, 61% per-agent)
```

Qwen3.5-35B-A3B (4 slots):
```
1 agent:  181 tok/s  (100% per-agent speed)
2 agents: 234 tok/s  (~117 tok/s each, 65% per-agent)
4 agents: 320 tok/s  (~80 tok/s each, 44% per-agent)
```

If your workflow has more concurrent agents than slots, requests queue until a slot is free. Consider multi-GPU routing (below) for higher concurrency.

### Context window usage

VRAM scales with context usage. The default RTX 5090 profiles are tuned for maximum context:

| Model | Default context | VRAM at idle | VRAM at full context |
|-------|----------------|--------------|---------------------|
| Qwen3-Coder-30B-A3B | 192K | 25.0 GB | ~28.9 GB |
| Qwen3.5-35B-A3B | 192K | 25.3 GB | ~26.1 GB |
| Hermes-4.3-36B | 32K | 24.5 GB | ~27.8 GB |

To reduce VRAM usage, lower the context window:

```bash
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  -e FOUNDRY_CTX_LENGTH=32768 \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

## Structured Output

All three models support JSON mode for structured outputs:

```python
response = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
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
    "model": "qwen3.5-35b-a3b",
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

Hermes-4.3-36B is specifically trained for tool calling with `<tool_call>` XML format. Qwen also supports tool calling via its chat template.

```python
response = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
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

Qwen3.5-35B-A3B supports a thinking mode where it shows its reasoning process in `<think>` tags before answering.

The server returns thinking content in the `reasoning_content` field:

```python
response = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{"role": "user", "content": "What is 127 * 389?"}],
    max_tokens=512,
)

msg = response.choices[0].message
if hasattr(msg, "reasoning_content") and msg.reasoning_content:
    print(f"Thinking: {msg.reasoning_content}")
print(f"Answer: {msg.content}")
```

Hermes-4.3-36B also supports thinking mode via the `<think>` tag convention. Enable it by including a system prompt that triggers deep reasoning (see the model's chat template).

## Streaming

All endpoints support Server-Sent Events (SSE) streaming for real-time token delivery:

```python
stream = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{"role": "user", "content": "Write a poem about GPUs."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

The Docker Compose configuration includes TCP tuning (BBR congestion control, busy polling) for minimal streaming latency. Time-to-first-token is typically ~50-200 ms depending on prompt length.

## Multi-GPU Agent Routing

For workloads requiring more concurrent agents than slots, run multiple Foundry instances and load-balance across them.

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

# Route based on agent ID (adjust slots_per_gpu to match your model's --parallel setting)
slots_per_gpu = 3  # Qwen3-Coder default; use 4 for Qwen3.5/Hermes
gpu_endpoints = [
    "http://localhost:8080/v1",  # GPU 0
    "http://localhost:8081/v1",  # GPU 1
]

def get_client(agent_id: int) -> OpenAI:
    endpoint = gpu_endpoints[agent_id // slots_per_gpu]
    return OpenAI(base_url=endpoint, api_key="sk-local")
```

## Troubleshooting

### Model loads on CPU instead of GPU

If you see `no devices with dedicated memory found` in the logs, the CUDA backend failed to load. Check:

1. **NVIDIA driver**: `nvidia-smi` should work on the host
2. **Container GPU access**: `docker run --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi`
3. **CUDA libraries**: The image must contain `libcudart.so.12`, `libcublas.so.12`, and `libcublasLt.so.12` in `/app/`

### Server responds slowly or hangs

1. Check if all layers are on GPU: look for `offloaded N/N layers to GPU` in container logs
2. Check VRAM: `nvidia-smi` -- if VRAM is full, reduce context with `FOUNDRY_CTX_LENGTH`
3. Check if all slots are occupied: `curl http://localhost:8080/metrics | grep slots`

### Connection refused

1. Container might still be loading the model. Check `docker logs <container>` for progress.
2. First run downloads the model (~20 GB) which can take several minutes.
3. Port conflict: use `-p 8081:8080` to map to a different host port.

### Out of VRAM

Reduce context window or switch to a smaller quantization:

```bash
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/foundry:/models \
  -e FOUNDRY_CTX_LENGTH=16384 \
  ghcr.io/infernet-org/foundry/qwen3.5-35b-a3b:latest
```

For GPUs with less than 24 GB VRAM, use Qwen (MoE) -- its expert offloading can spill inactive experts to CPU.

### Inconsistent response speeds

If response speed varies between requests, check for:

1. **Prompt cache misses**: First message in a conversation is always slower (prompt processing)
2. **Concurrent slot contention**: Other agents may be using slots simultaneously
3. **GPU thermal throttling**: Check `nvidia-smi -q -d PERFORMANCE` for throttle reasons
4. **CPU interrupt interference**: Pin GPU IRQs to dedicated cores (see README Host Tuning section)
