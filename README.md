# l3mcore — Light Easy Mix Of Experts
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/V5E82133QQ)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/lemoelink/l3mcore)
> [!WARNING]
> **Fase de Desarrollo / In Development**
> Este proyecto se encuentra actualmente en fase de desarrollo activo. Sin embargo, su funcionalidad principal (enrutamiento, carga de modelos y backends) es completamente operativa y estable para su uso.
>
> This project is currently under active development. However, its core functionality (routing, model loading, and backends) is fully operational and stable for use.

A lightweight Mixture of Experts (MoE) system that acts as an intelligent router for AI requests. It exposes an API fully compatible with OpenAI and Ollama, classifies user input, and automatically dispatches it to the appropriate expert model.

> Part of the **[lemoe.link](https://lemoe.link)** ecosystem.

###  Watch it in action!
https://github.com/user-attachments/assets/e97e1481-a0a3-4f25-a3de-7ed25936e2b3

## Table of Contents

- [l3mcore — Light Easy Mix Of Experts](#l3mcore--light-easy-mix-of-experts)
    - [ Watch it in action!](#-watch-it-in-action)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Manual Installation](#manual-installation)
  - [Docker Installation](#docker-installation)
    - [Available Tags](#available-tags)
    - [Docker Compose (with Open WebUI)](#docker-compose-with-open-webui)
  - [Configuration](#configuration)
    - [1. Router (config.json)](#1-router-configjson)
    - [2. Experts (experts.json)](#2-experts-expertsjson)
    - [3. Expert Backends](#3-expert-backends)
  - [Router Architecture](#router-architecture)
    - [Hybrid Scoring System](#hybrid-scoring-system)
    - [Scoring Weights Tuning](#scoring-weights-tuning)
    - [Fallback Chain](#fallback-chain)
  - [API Usage](#api-usage)
  - [Security](#security)
  - [Project Structure](#project-structure)

---

## Features

- **OpenAI and Ollama Compatible API**: Drop-in replacement for Ollama (port 11435) or an OpenAI-compatible server (`/v1/chat/completions`). Works with Open WebUI, Continue (VSCode), LiteLLM, and any OpenAI client.
- **Precision Hybrid Router**: Three-tier routing system:
  - *Semantic Embedding*: Multi-vector comparison using SentenceTransformers with 4-signal hybrid scoring and softmax normalization.
  - *ML Classifier*: Standard BERT/RoBERTa classification for ultra-fast routing when a fine-tuned model is available.
  - *Keyword Fallback*: Fuzzy matching algorithm (rapidfuzz) when AI methods are unavailable or below the confidence threshold.
- **Multi-Backend Expert Dispatcher**: Connects up to 15 simultaneous experts across different backends:
  - External APIs (OpenAI, Anthropic, Gemini, etc. via LiteLLM)
  - Local Ollama instances
  - Local ONNX or GGUF models (Llama.cpp)
- **Cascading Contextual Routing**: Stateless conversation-aware routing. When a user message is ambiguous (e.g. "Make it shorter"), the router evaluates the last 2-3 user messages from the conversation history to maintain topic continuity. No database or session state required — each request carries its own context via the standard OpenAI messages array. Configurable via `context_messages` and `context_max_chars` in `config.json`.
- **Silent Self-Correction**: If any expert fails (timeout, API down, connection refused), the system automatically and silently redirects the request to the fallback model. The user receives a normal response without knowing an error occurred. Administrators can monitor failures via `[Auto-Correction]` log entries.
- **Smart Memory Management**: Limits RAM usage for local models (maximum 3 simultaneous), with LRU eviction and TTL-based automatic unloading after 5 minutes of inactivity.
- **Rate Limiting**: Built-in per-IP sliding-window rate limiter (60 req/min by default). Supports `X-Forwarded-For` for reverse proxy setups.
- **Request Size Protection**: Incoming request bodies capped at 1 MB to prevent memory exhaustion.
- **Health Endpoint**: `GET /health` reports the live status of every core component (router mode, models in memory, plugins loaded, degraded state detection).
- **Routing Diagnostic Endpoint**: `GET /v1/route?text=<prompt>` runs the router against any text and returns the full scoring breakdown — expert selected, confidence score, and top-5 ranked alternatives. No model is invoked; useful for configuration and debugging.
- **Extensible Architecture (Plugins & Tools)**: 
  - *Plugins*: Python modules in the `plugins/` directory that hook into the core lifecycle (`override_route`, `before_routing`, `after_generation`) to modify behavior seamlessly.
  - *Tools*: Standalone Python scripts in the `tools/` directory that act as callable functions for models with tool-calling capabilities. Just drop a `.py` file in the folder and the `tools` expert will automatically register it.

---

## Quick Start

Requires Python 3.10 or higher. The easiest way to install and configure l3mcore is using our interactive auto-installer, which will download the repository, check dependencies, and set up the virtual environment in one step.

**Using wget:**
```bash
wget -qO- https://raw.githubusercontent.com/lemoelink/l3mcore/refs/heads/master/setup.sh | bash
```

**Using curl:**
```bash
curl -sSL https://raw.githubusercontent.com/lemoelink/l3mcore/refs/heads/master/setup.sh | bash
```

*(Note: The setup script will automatically create and activate an isolated Python virtual environment `venv` to install all requirements securely without breaking your system packages).*

Once the setup finishes, start the server:

```bash
cd LeMoE
./start.sh
```

The server will be available at `http://0.0.0.0:11435`.

### Classic Git Clone

If you prefer to clone the repository manually:

```bash
git clone https://github.com/lemoelink/l3mcore.git
cd l3mcore
./setup.sh
./start.sh
```

---

## Manual Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Docker Installation

l3mcore is available on Docker Hub and can be run entirely via Docker, avoiding the need to install Python dependencies on your host.

```bash
# Run the default image (Debian Slim, CPU)
docker run -d -p 11435:11435 \
  -v ./config:/app/config \
  -v ./models:/app/models \
  -v ./data:/app/data \
  --name l3mcore \
  lemoelink/l3mcore:latest
```

### Available Tags

| Tag | Base | Use case |
|---|---|---|
| `lemoelink/l3mcore:latest` | Debian Slim | Default. CPU inference, smallest footprint. |
| `lemoelink/l3mcore:debian` | Debian Standard | Same as `latest` but with full OS libraries. |
| `lemoelink/l3mcore:cuda` | Nvidia CUDA *(Beta)* | GPU acceleration (Nvidia). Requires `--gpus all`. |
| `lemoelink/l3mcore:rocm` | AMD ROCm *(Beta)* | GPU acceleration (AMD). Requires mounting host DRI devices. |

**GPU Acceleration details:**
- **Nvidia CUDA**: Run with `--gpus all` (or specify individual GPUs).
- **AMD ROCm**: Run with `--device=/dev/kfd --device=/dev/dri --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined`.

**Volume Mounts:**
- `/app/config`: Maps your `config.json` and `experts.json` so you can edit routing rules on the fly without rebuilding.
- `/app/models`: Persists downloaded GGUF/ONNX models and HuggingFace caches across container restarts.
- `/app/data`: Persists LRU/TTL runtime metrics.

### Docker Compose (with Open WebUI)

The easiest way to get a full stack (l3mcore + Open WebUI chat interface) running is via the interactive setup script:

```bash
bash private/docker/setup-docker.sh
docker compose up -d
```

The script generates a `docker-compose.yml` with:
- **`l3mcore`** service on port `11435`
- **`open-webui`** service on port `3000`, pre-configured to talk to l3mcore

Once running:
- Open WebUI → `http://localhost:3000`
- l3mcore API → `http://localhost:11435`

To rebuild and push all images to Docker Hub:

```bash
bash private/publish_docker.sh
```

---

## Configuration

All configuration lives in the `config/` folder.

### 1. Router (config.json)

The file `config/config.json` controls the routing engine.

```json
{
    "router": {
        "mode": "generic",
        "router_type": "embedding",
        "model_path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "categories_file": "config/experts.json",
        "confidence_threshold": 0.4,
        "confidence_threshold_keyword": 0.3,
        "keyword_fallback": true,
        "softmax_temperature": 0.15,
        "scoring_weights": {
            "max_keyword":  0.40,
            "description":  0.30,
            "mean_keyword": 0.20,
            "top3_vote":    0.10
        }
    },
    "ai_engine": {
        "n_threads": 4,
        "n_ctx": 2048,
        "n_batch": 512
    },
    "expert_runner": {
        "api_timeout": 60,
        "ollama_timeout": 60,
        "ollama_allowed_hosts": ["localhost", "127.0.0.1"]
    }
}
```

| Field | Type | Description |
|---|---|---|
| `mode` | string | `generic` uses `experts.json`. `model` uses a trained classifier. |
| `router_type` | string | `embedding` (SentenceTransformers) or `classification` (fine-tuned BERT). |
| `model_path` | string | HuggingFace repo or local path to the router model. Empty string disables AI routing and uses keyword-only mode. |
| `categories_file` | string | Path to `experts.json`. Must stay within the project directory. |
| `confidence_threshold` | float | Minimum score (0-1) to accept a router ML prediction. |
| `confidence_threshold_keyword` | float | Minimum score (0-1) for the keyword/fuzzy fallback tier. Defaults to `confidence_threshold` if not set. |
| `keyword_fallback` | bool | Enable keyword + fuzzy matching as secondary routing method. |
| `softmax_temperature` | float | Controls sharpness of softmax normalization. Lower = more decisive (try 0.10-0.20). |
| `scoring_weights` | object | Hybrid scoring signal weights. See [Scoring Weights](#scoring-weights-tuning). |
| `context_messages` | int | Number of recent user messages to concatenate for cascade routing. Default: `3`. |
| `context_max_chars` | int | Maximum characters for concatenated context text. Default: `1600`. |
| `ai_engine.n_threads` | int | CPU threads for GGUF inference (llama.cpp). Default: `4`. |
| `ai_engine.n_ctx` | int | Context window size for the GGUF model. Default: `2048`. |
| `expert_runner.api_timeout` | int | Seconds before an external API call times out. Default: `60`. |
| `expert_runner.ollama_allowed_hosts` | list | Hostnames allowed for Ollama endpoints. `localhost` and `127.0.0.1` always included. |

**Recommended router model:**

- **Multilingual (Default)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (excellent for Spanish, English, and other languages)

#### Autonomous Keyword Enrichment

L3MCOre includes a built-in semantic keyword enrichment system. On startup or configuration reloads, a background thread automatically uses the fallback LLM (or any active Ollama/API expert) to generate 20 relevant synonyms and query patterns for each category based on its name, description, and base keywords. The enriched keywords are cached locally in `config/.experts_enriched_cache.json` for zero-latency boots, automatically improving the semantic router's alignment without requiring the user to manually compile massive keyword lists.

---

### 2. Experts (experts.json)

Defines the list of specialist models the router dispatches to.

```json
{
  "max_experts": 16,
  "experts": [
    {
      "id": 0,
      "label": "fallback",
      "description": "General fallback model used when the router cannot classify the request.",
      "type": "local",
      "format": "gguf",
      "model_path": "models/Qwen3.5-0.8B-Q4_K_M.gguf"
    },
    {
      "id": 1,
      "label": "programador",
      "description": "Expert in writing and reviewing code in any programming language",
      "keywords": ["codigo", "programar", "python", "javascript", "funcion",
                   "script", "error", "bug", "html", "css", "clase", "objeto",
                   "modulo", "api", "refactorizar"],
      "type": "ollama",
      "url": "http://127.0.0.1:11434",
      "model_name": "qwen2.5:7b"
    }
  ]
}
```

**IMPORTANT: The Fallback Expert**
The `fallback` expert (ID `0`) is mandatory and must not have any `keywords`. When the router's confidence falls below the `confidence_threshold`, or no keywords match, the system routes the request to this fallback expert. You can configure this fallback to use any backend (local GGUF, Ollama, API) just like any other expert.

**Keyword quality is critical.** The embedding router builds individual semantic vectors for each keyword and combines them using a 4-signal scoring formula. A minimum of **15 descriptive keywords per expert** is required for reliable routing. With fewer keywords the `mean_keyword` and `top3_vote` signals are too sparse and accuracy drops.

Write keywords as concrete, domain-specific terms the user would actually say — not abstract categories. For example, for a coding expert use `"funcion"`, `"clase"`, `"loop"`, `"variable"` rather than just `"programacion"`.

---

### 3. Expert Backends

Each expert entry in `experts.json` supports three `type` values:

**`ollama`** — Local or remote Ollama instance:
```json
{
  "type": "ollama",
  "url": "http://127.0.0.1:11434",
  "model_name": "qwen2.5:7b"
}
```

**`api`** — External API via LiteLLM (OpenAI, Anthropic, Gemini, etc.):
```json
{
  "type": "api",
  "provider": "openai",
  "model_name": "gpt-4o-mini",
  "api_key_env": "OPENAI_API_KEY"
}
```
The API key is read from the environment variable named in `api_key_env`. Never put keys directly in the JSON file.

**`local`** — Local ONNX or GGUF model:
```json
{
  "type": "local",
  "format": "onnx",
  "label": "my_model",
  "model_path": "models/my_model"
}
```

---

## Router Architecture

### Hybrid Scoring System

The default `embedding` router builds three data structures per expert at startup and combines four signals at inference time.

**At startup** (`_precompute_category_embeddings`):

```
Expert "programador"
   kw_vecs  : [embed("codigo"), embed("python"), embed("script"), ...]  ← one vector per keyword
   centroid : L2-normalised mean of all keyword vectors
   desc_vec : embed("Expert in writing and reviewing code...")
```

**At inference** (`_embed_score`):

The user query is encoded as `"query: <user text>"` and compared against each expert's data using four signals:

| Signal | Weight | What it captures |
|---|---|---|
| `max_keyword` | 40% | Maximum cosine similarity between the query and any single keyword. High when the user uses an exact domain term. |
| `description` | 30% | Cosine similarity between the query and the full expert description. Captures intent that keywords alone might miss. |
| `mean_keyword` | 20% | Average similarity across all keywords. Measures overall domain alignment. |
| `top3_vote` | 10% | Fraction of the top-3 keyword scores that exceed 0.40. Penalises lucky single-keyword matches and rewards consensus. |

After computing a raw hybrid score for every expert, the scores are passed through **softmax normalization** so the final score is a real probability (0-1) rather than a raw cosine value. This makes the `confidence_threshold` meaningful regardless of the embedding model used.

```
raw_scores: {programador: 0.61, sysadmin: 0.53, traductor: 0.41, ...}
    ↓ softmax(temperature=0.15)
norm_scores: {programador: 0.82, sysadmin: 0.14, traductor: 0.02, ...}
    → winner: "programador" at 0.82
```

### Scoring Weights Tuning

You can adjust weights in `config.json` without touching code:

```json
"scoring_weights": {
    "max_keyword":  0.40,
    "description":  0.30,
    "mean_keyword": 0.20,
    "top3_vote":    0.10
}
```

Tuning guidelines:
- **Raise `max_keyword`** if users tend to use exact technical terms.
- **Raise `description`** if expert descriptions are detailed and precise.
- **Lower `softmax_temperature`** (toward 0.05) to make the router more decisive when experts are clearly distinct.
- **Raise `softmax_temperature`** (toward 0.30) if experts overlap and you want smoother transitions.

### Fallback Chain

```
User query
    
    
Cascade Step 1: Evaluate last user message
     score < confidence_threshold
    
Cascade Step 2: Evaluate last 2-3 user messages (concatenated)
     score < confidence_threshold
    
Keyword + Fuzzy matching (rapidfuzz)
     score < confidence_threshold_keyword
    
Fallback expert (ID 0)
```

---

## API Usage

The server starts on `http://0.0.0.0:11435` by default.

**OpenAI-compatible endpoint:**
```bash
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Escribe un script en Python para listar archivos"}],
    "stream": false
  }'
```

**Route directly to a specific expert** (skips the router):
```bash
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "programador",
    "messages": [{"role": "user", "content": "Refactoriza esta funcion"}]
  }'
```

**Ollama-compatible endpoints:**
```bash
# List available models
curl http://localhost:11435/api/tags

# Chat
curl -X POST http://localhost:11435/api/chat \
  -d '{"model": "l3mcore", "messages": [{"role": "user", "content": "Hola"}]}'
```

**Health check:**
```bash
curl http://localhost:11435/health
```

**Available routes:**
| Method | Path | Description |
|---|---|---|
| GET | `/` | Server info |
| GET | `/health` | Live status of all core components |
| GET | `/v1/models` | List models (OpenAI format) |
| POST | `/v1/chat/completions` | Inference (OpenAI format, streaming supported) |
| GET/POST | `/v1/route` | **Routing diagnostic** — scores a text against all experts without generating a response |
| GET | `/v1/discover` | Discovers available Ollama models not yet configured as experts |
| GET | `/v1/health/experts` | Health status of each individual expert |
| GET | `/api/tags` | List models (Ollama format) |
| POST | `/api/chat` | Inference (Ollama format, streaming supported) |
| GET | `/api/version` | Server version |

---

## Security

l3mcore includes hardened defaults for production-adjacent use:

| Protection | Details |
|---|---|
| **Path Traversal** | Model paths and labels are canonicalized and confined to the `models/` directory. Labels containing `/`, `\`, or `..` are rejected. |
| **SSRF Protection** | Ollama URLs are validated: only `http`/`https` schemes accepted. Cloud metadata endpoints (`169.254.0.0/16`) are blocked. Hostname allowlist configurable via `ollama_allowed_hosts`. |
| **Request Size Limit** | Incoming request bodies are capped at 1 MB (`MAX_CONTENT_LENGTH`). |
| **Rate Limiting** | Sliding-window rate limiter: 60 requests per minute per IP. Respects `X-Forwarded-For` when behind a reverse proxy. Returns HTTP 429 when exceeded. |
| **Log Sanitization** | User input is stripped of control characters and ANSI escape sequences before being written to log files. |
| **Config Path Validation** | `categories_file` paths in `config.json` are resolved and must stay within the project directory. |
| **Silent Error Isolation** | Internal exceptions from expert backends are never exposed to the end user. Error details are logged server-side only. The user receives a generic fallback response. |
| **Atomic File Writes** | Model stats are written atomically (write to `.tmp`, then `os.replace()`) to prevent corruption on crash. |
| **API Timeouts** | All external calls (LiteLLM, Ollama) have a configurable timeout (default 60 s) to prevent hung requests. |
| **experts.json Validation** | Schema validation at startup: missing required fields per expert type are logged immediately, not at first request. |

---

## Project Structure

```text
l3mcore/
├── api_server.py          # Flask API server (OpenAI + Ollama compatible)
├── main.py                # CLI entry point (stdin loop)
├── setup.sh               # Interactive setup: Ollama check + model download
├── start.sh               # Starts the API server (creates venv if needed)
├── requirements.txt       # Python dependencies
├── config/
│   ├── config.json        # Router and server configuration
│   └── experts.json       # Expert definitions (labels, keywords, backends)
├── models/                # Local ONNX or GGUF model directories
├── data/                  # Runtime data (model usage stats)
├── logs/                  # Application logs
├── plugins/               # Plugin directory (Python lifecycle hooks)
├── tools/                 # Tool directory (Callable functions for LLMs)
├── private/
│   └── docker/
│       ├── Dockerfile         # Default Debian Slim (CPU)
│       ├── Dockerfile.debian  # Standard Debian (CPU)
│       ├── Dockerfile.cuda    # Nvidia CUDA (GPU)
│       └── setup-docker.sh    # Interactive Docker Compose generator
├── tests/
│   └── test_adversarial.py  # Adversarial test suite
└── modules/
    ├── generic_router.py  # Hybrid embedding/classification router
    ├── decision_router.py # Fine-tuned classifier router (model mode)
    ├── router_factory.py  # Router selection factory
    ├── utils_router.py    # Shared router utilities (clean_text, model loader)
    ├── expert_runner.py   # Expert dispatcher (API / Ollama / Local)
    ├── onnx_runner.py     # ONNX model runner with LRU + TTL memory management
    ├── ai_engine.py       # GGUF fallback engine (llama-cpp-python)
    ├── config_manager.py  # JSON config loader (with disk-change detection)
    ├── plugin_manager.py  # Plugin loader and hook dispatcher
    └── logger.py          # Shared application logger
```
