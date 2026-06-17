# Changelog

All notable changes to this project will be documented in this file.

Format: Keep a Changelog — https://keepachangelog.com/en/1.0.0/
Versioning: Semantic Versioning — https://semver.org/spec/v2.0.0.html

---

## [Unreleased]

### Added

- `modules/config_manager.py`: Added dynamic, recursive de-obfuscation in `_deobfuscate_value` supporting `env:`, `base64:`, and `obfuscated:` prefixes, applied on-the-fly when calling `get()` or `get_all()`.

- `api_server.py`: Added `_clean_assistant_response()` helper function to parse and strip any raw JSON tool-calling blocks from the final conversational response.

- `tools/paperless_search.py`: Added automatic bilingual query translation, mapping Spanish business terms (e.g., *factura*, *nómina*, *alquiler*) to English to enable cross-language fallback searches in Paperless-ngx.

- `tools/paperless_search.py`: Added context pruning utility (`_prune_old_contexts`) to dynamically remove previous document context from the chat history, preventing token inflation and context confusion.

- `plugins/image_router.py`: Added support for Ollama-style `images` message payload schema (multimodal inputs) to properly route vision queries to the vision expert.

- `private/test_image_router.py`: Added unit tests for Ollama vision format handling and updated assertions to match Spanish output logs.

- `modules/logger.py`: Added a custom `ColorFormatter` to color-code console log outputs without using emojis (Green for INFO, Yellow for WARNING, Red for ERROR/CRITICAL).

- `modules/generic_router.py`: Bypassed the minimum keywords check for the fallback expert (ID 0 or label 'fallback') to avoid unnecessary validation warnings during startup.

- `start.sh` & `setup.sh`: Replaced references to `LEMoE` with `L3MCOre`, cleaned up symbols, and applied ANSI green/yellow formatting to startup information.

- `plugins/telemetry_dashboard.py`: Removed unicode arrow `→` and colorized the server startup output message.

- `modules/expert_runner.py`: `_inject_system_prompt()` helper. Reads `system_prompt` from the expert config dict and prepends it as a `{"role": "system"}` message before each inference call. Truncated to 4000 chars. Never mutates the original messages list. If a system message from the client already exists it is preserved after the expert prompt, not discarded.

- `modules/expert_runner.py`: latency measurement in `ExpertDispatcher.run()`. Uses `time.monotonic()` around the backend call and reports `latency_ms` to the telemetry plugin via `_notify_latency()` side-channel after a successful response.

- `api_server.py`: `_fire_failure_webhook()`. When any expert fails and the system auto-corrects to fallback, a best-effort POST is sent to the URL configured in `config.json > expert_runner.failure_webhook_url`. Payload: `{"event": "expert_failure", "expert": "<label>", "reason": "<msg>"}`. Timeout: 3s. Any error is logged and swallowed — the fallback chain is never interrupted.

- `api_server.py`: `GET /v1/discover` endpoint. Queries a local (or remote) Ollama instance and returns the list of installed models that are not yet configured as experts, along with a ready-to-paste `experts.json` snippet for each. Query param: `url` (default `http://127.0.0.1:11434`). Labels are sanitized with `re.sub` before being included in the response.

- `plugins/telemetry_dashboard.py`: `record_latency(label, latency_ms)` public function. Called by the core's `_notify_latency` before `after_generation` fires. Stores the value in `_pending_latency` (thread-safe) so `after_generation` can consume it atomically in the same request cycle.

- `plugins/telemetry_dashboard.py`: per-expert average latency and estimated API cost columns in the dashboard UI. Latency is computed as `latency_sum_ms / requests` client-side. Cost is computed from a price table (`_COST_PER_1M`) only for `type: api` experts; local and Ollama experts always show `-`.

- `plugins/telemetry_dashboard.py`: `_COST_PER_1M` price table covering GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo, Claude 3.5 Sonnet, Claude 3 Haiku, Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash.

- `plugins/routing_transparency.py`: new official plugin. Appends a configurable footer to every response showing which expert handled the request and the router's confidence score. All config via `config.json > routing_transparency`. Expert label sanitized against `^[a-zA-Z0-9_\-]{1,64}$` before inclusion in output.

- `api_server.py`: `_start_keyword_enrichment()` background task. Automatically queries the first available LLM expert on startup or reload to generate 20 synonyms and query patterns for each category. Saves results to `config/.experts_enriched_cache.json` using a SHA-256 hash of `experts.json` to enable instant 0 ms subsequent boots.

- `api_server.py`: `_apply_enriched_keywords()`. Safely merges enriched keywords and recomputes the semantic router embeddings at runtime under a thread-safe `_cache_lock`.

- `api_server.py`: `GET/POST /v1/route` diagnostic endpoint. Runs the semantic router against any text and returns expert, score, method, cascade step and top-5 breakdown. No model is invoked.

### Changed

- `api_server.py`: Bypassed the classic `hook_override_route` execution when `tool_calling.enabled` is `true` in configuration, preventing query duplication and prompt pollution.

- `tools/bi_reporter.py`: Added `include_sql` flag to `_format_results()` and configured it to `False` in `execute_tool()`, preventing internal SQL queries from being shown in tool-calling results.

- `config/experts.json`: Refined the system prompt of the `tools` expert to explicitly restrict outputting internal queries or technical debug details.

- `config/config.json`: Obfuscated Odoo database password using base64 format under the `erp_connector` configuration block.

- `config/experts.json`: Configured a dedicated `system_prompt` for `document-expert` and `image-expert`, specifically authorizing the LLM to access and analyze personal and financial data to bypass safety guardrail refusals on user-owned documents.

- `tools/paperless_search.py`: Enhanced intent detection (`_is_retrieval_intent`) to cover direct factual query patterns, with a fallback heuristic check if the BERT intent classifier returns a negative score.

- `modules/expert_runner.py`: Replaced `_notify_latency` with `_notify_telemetry` to pass detailed token counts (estimated for local ONNX experts, actual for API and Ollama experts) and execution success status to the telemetry plugin.

- `api_server.py`: Modified fallback routing (`_do_fallback`) to direct unrecognized follow-up queries to `document-expert` when an active document context is present in the chat history.

- `README.md`: Updated automated setup script URLs to point to the correct `l3mcore` repository path.

- `config/config.json`: Changed default embedding model to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to support unified multilingual routing.

- `setup.sh`: Replaced the choice of downloading either English or Spanish models with a single Y/n prompt for the multilingual router model, saving `""` (disabled) to `config.json` if declined.

- `api_server.py`: `_run_inference` now calls `_fire_failure_webhook(label, reason)` on every `[Auto-Correction]` event (phases 0, 1, and 2).

- `plugins/telemetry_dashboard.py`: `_update_telemetry` now accepts `latency_ms` and `cost_usd` parameters and persists them in `telemetry.json` alongside `requests` and `tokens`.

- `plugins/telemetry_dashboard.py`: rewrote auto-refresh from `window.location.reload()` every 5s to `fetch('/api/stats')` + DOM patch. Page state (scroll, open modals) is never interrupted.

- `plugins/telemetry_dashboard.py`: dashboard now shows 7 columns: Expert Router, Reqs, Tokens (In/Out), Throughput, Avg Latency, Cost, Health.

### Added (Paperless RAG)

- `tools/paperless_search.py`: Native l3mcore tool implementing `override_route` hook for dynamic RAG integration with Paperless-ngx. Features include Unicode normalization for robust accent-insensitive query parsing, a double-layer intent validator (obligatory custom BERT intent classifier and optional fallback semantic MoE router), heuristic query cleaning, and dual-URL resolution (`api_url` and `web_url`) for secure local network and Docker deployments.

- `tools/paperless_search.py`: Automatic model downloader and updater thread that verifies, downloads, and reloads the custom BERT sequence classification model from Hugging Face (`lemoelink/LEMoEPPC-onnx`) upon plugin initialization and performs periodic update checks every 7 days (or on system boot) cleanly in a background daemon.

- `private/test_paperless_search.py`: A comprehensive, isolated test suite for the Paperless search plugin. Validates configuration precedence, Unicode accent normalizations, search query stop-word/verb cleaning, intent classification checks, custom BERT ONNX classification mock inferences, mock API requests, and graceful timeout/error handling.

- `private/uso_del_plugin.md`, `private/uso_del_plugin_desarrolladores.md`, `private/funcionamiento_del_plugin.md`, `private/explicacion_tecnica_y_basica.md`, `private/11_Propuesta_Destilacion_Consultas.md`: Extensive Spanish documentation files covering final user instructions, development configuration details, lifecycle request flows, architectural specs with diagrams, and the technical proposal for Query Reformulation & Distillation.

- `private/english_guide.md`: Fully detailed English integration guide covering O(1) constant-time stop-word performance analysis, JSON configurations, conversational phrasing, and Mermaid cleaning diagrams.

### Fixed

- `tools/paperless_search.py`: Implemented a query synonym expander `_expand_synonyms` to map "dron" / "drones" to `(dron OR drones OR UAS)` inside the search queries, preventing search failures due to vocabulary mismatch.

- `tools/paperless_search.py`: Stripped copy-pasted markdown formatting, bullet prefixes, and backticks from the user's input before running intent classification and search.

- `tools/paperless_search.py`: Added "cv", "curriculum", and related variants to the trigger keywords list and Spanish-to-English translation dictionary.

- `modules/expert_runner.py`: Sanitized and formatted image message payloads. Ollama now receives raw base64 strings (stripping `data:image/...;base64,` prefixes) and LiteLLM/APIs receive correct data-URI formatted image strings.

- `config/experts.json`: Fixed `image-expert` model tag from `llava` to `llava:7b` to match the exact model name installed in local Ollama.

- `docs/src/css/custom.css`: Fixed mobile sidebar menu visibility by restricting the navbar's `backdrop-filter: blur` styling to desktop viewports.

- `tools/paperless_search.py`: Optimized the injected `system_instruction` prompt template to explicitly and obligatorily require the LLM to output a concise, structured executive summary.

- `tools/paperless_search.py`: Implemented a safeguard to ignore background task prompts sent by the frontend UI, preventing recursive plugin executions and API timeout crashes.

---

## [0.4.0] - 2026-06-04

### Added

- `docker/Dockerfile.rocm`: Added native AMD ROCm 6.1.2 Dockerfile configuration for building l3mcore with AMD GPU hardware acceleration.
- `private/publish_docker.sh` and `docker/Dockerfile*` suite: Published new container images under the `lemoelink/l3mcore` namespace on Docker Hub:
  - `lemoelink/l3mcore:latest` (Debian Slim CPU default image, optimized footprint)
  - `lemoelink/l3mcore:debian` (Full Debian image for CPU)
  - `lemoelink/l3mcore:cuda` (Ubuntu base with Nvidia CUDA 12.1 and pre-compiled CUDA wheels for GPU acceleration)
  - `lemoelink/l3mcore:rocm` (AMD ROCm 6.1.2 base with compiled HIP/ROCm acceleration for AMD graphics cards)
- `README.md`: Updated documentation detailing the available Docker tags and commands required to map GPU accelerators.
- `api_server.py`: Background thread monitoring daemon `_start_experts_watcher` that polls the modification time (`mtime`) of `config/experts.json` every 2 seconds. Automatically triggers a seamless hot-reload upon file saving or direct modification by administrators.
- `api_server.py`: Class-level `reload_experts` method inside the `_Core` singleton, protected by `threading.Lock` for atomic runtime reloads of enrutables and available models.
- `modules/generic_router.py`: `reload_categories` method in `GenericRouter` to allow live reconstruction of enroutable categories and SentenceTransformer multi-vector representations at runtime, clearing prediction cache cleanly.
- `plugins/system_time.py`: Lightweight plugin that injects the current local system date and time as a system message at the beginning of the messages list.
- `private/test_system_time.py`: Unit test suite for the `system_time.py` plugin.
- `plugins/user_profile.py`: Lightweight plugin that injects the user's name, preferences, and custom instructions as a system message at the beginning of the conversation.
- `private/test_user_profile.py`: Unit test suite for the `user_profile.py` plugin.

### Changed

- `config/experts.json`: Tailored the entire list of 17 experts to run on low-resource CPU-only systems. GGUF-based and API-based experts migrated to local Ollama endpoints with lightweight models: `qwen2.5:1.5b` and `llama3.2:1b`.
- `start.sh`: Added a dynamic `.env` file parser to automatically load and export local environment variables upon server execution.
- `setup.sh`: Translated Spanish update checks and output logs to English, added prerequisites verification checks for a C/C++ compiler and `make` utility.

### Fixed

- `setup.sh`: Implemented auto-detection and auto-cleanup of corrupted or incomplete virtual environment directories.
- `setup.sh` and `start.sh`: Resolved a false-positive update notification by dynamically checking the current local Git branch name.

---

## [0.3.0] - 2026-05-29

### Added

- `modules/plugin_manager.py`: class-level `threading.Lock` for the singleton `__new__`. Prevents double initialization under concurrent access.
- `modules/plugin_manager.py`: filename validation for plugin files. Each `.py` filename checked against `^[a-zA-Z0-9_-]+$` before loading.
- `modules/plugin_manager.py`: isolated plugin namespace in `sys.modules`. Each plugin module registered under `lemoe_plugin.<name>`.
- `modules/plugin_manager.py`: pre-computed `after_generation` signature flag. The `hook_after_generation` hot path no longer calls `inspect.signature` on every request.
- `modules/config_manager.py`: class-level `threading.Lock` for the singleton `__new__`.
- `modules/generic_router.py`: `_cache_lock = threading.Lock()` instance attribute to protect `_predict_cache` under concurrent access.
- `modules/decision_router.py`: same `_cache_lock` pattern as `generic_router`.

### Changed

- `modules/plugin_manager.py` `hook_before_routing`: return value from each plugin is type-checked before being assigned to `prompt`. Non-string values preserved.
- `modules/plugin_manager.py` `hook_after_generation`: same type guard as `hook_before_routing`.
- `modules/config_manager.py` `save()`: atomic write via `.tmp` + `os.replace()`.
- `modules/config_manager.py`: `CONFIG_FILE` moved to module-level constant `_CONFIG_FILE`.
- `modules/generic_router.py` `_model_predict`: softmax computation changed to numerically stable form: `exp((s - max_raw) / temp)`.
- `modules/ai_engine.py` `_ensure_model_loaded`: removed unprotected read of `self.llm` outside the lock. Double-checked locking pattern formally correct.
- `setup.sh`: removed beta warning messages for the plugin system.

### Fixed

- PLG-LOAD-1: plugin double-registration under concurrent startup.
- PLG-LOAD-2: stdlib name collision via plugin filename.
- PLG-LOAD-3: plugin namespace isolation in sys.modules.
- PLG-LOAD-4: None propagation in hook_before_routing corrupting the prompt.
- PLG-LOAD-5: None propagation in hook_after_generation causing HTTP 500.
- PLG-LOAD-6: inspect.signature called on every request in hot path.
- CFG-1: non-atomic config.json write causing file corruption on crash.
- CFG-2: mutable class attribute allowing external config path mutation.
- GR-1: unsynchronized cache dict in GenericRouter under concurrent access.
- GR-2: softmax overflow producing nan routing scores at low temperatures.
- DR-1: unsynchronized cache dict in DecisionRouter under concurrent access.
- AI-1: unprotected read of self.llm outside lock in _ensure_model_loaded.

### Documentation

- `private/09_Auditoria_Core_v030.md`: full technical write-up of all findings, root causes, and corrections applied in this release.

---

## [0.2.0] - 2026-05-29

### Added

- `plugins/image_router.py`: inspection window limits. Only the last 10 messages and the first 20 content parts per message are scanned.
- `plugins/image_router.py`: strict type guards throughout the message-iteration loop.
- `plugins/image_router.py`: validation of the `image_url.url` field (data-URI or HTTP/HTTPS URL).
- `plugins/image_router.py`: data-URI format validation via compiled regular expression `_DATA_URI_RE`.
- `plugins/image_router.py`: base64 payload size warning (>524288 chars).
- `plugins/image_router.py`: expert label format validation against `^[a-zA-Z0-9_-]{1,64}$`.
- `private/08_Plugin_Image_Router.md`: technical documentation for the image_router plugin.

### Changed

- `plugins/image_router.py`: module-level docstring and inline comments removed. Code style aligned with the rest of the codebase.
- `plugins/image_router.py`: detection logic extracted into `_check_image_part()` helper.

### Security

- PLG-1: expert label validated against an alphanumeric pattern before use.
- PLG-2: message and part inspection bounded to prevent DoS from large payloads.
- PLG-3: image URL field validated for existence, type, and scheme.
- PLG-4: oversized base64 payloads generate a log warning.
- PLG-5: explicit isinstance checks prevent AttributeError on malformed message elements.

---

## [0.1.0] - Initial release

### Added

- Core l3mcore router with semantic E5 embeddings and cascading fallback logic.
- OpenAI-compatible API: GET /v1/models, POST /v1/chat/completions (streaming).
- Ollama-compatible API: GET /api/tags, POST /api/chat, GET /api/version.
- Plugin system with hooks: override_route, before_routing, after_generation.
- In-memory sliding-window rate limiter (100 req/60 s per IP).
- 1 MB request body cap (SEC-3).
- Security response headers: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy.
- SSRF protection in expert_runner.py: scheme validation and blocked network ranges for Ollama URLs (SEC-2).
- Log sanitization: control characters stripped before writing user input to application logs (SEC-7).
- Plugin: image_router.py — initial functional version without security hardening.
