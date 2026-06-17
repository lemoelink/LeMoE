"""
l3mcore API Server

OpenAI- and Ollama-compatible HTTP API. Any client that speaks either protocol
can use l3mcore as a drop-in backend by pointing its base URL to this server.

Endpoints:
  GET  /                    -> Server info
  GET  /v1/models           -> List available experts (OpenAI format)
  POST /v1/chat/completions -> Inference (OpenAI format, streaming supported)
  GET  /api/tags            -> List models (Ollama format)
  POST /api/chat            -> Inference (Ollama format, streaming supported)
  GET  /api/version         -> Server version
"""

import json
import os
import re
import time
import uuid
import threading
from collections import defaultdict
from collections import deque

# Ensure cwd is always the script directory
# (necessary for relative model paths to resolve correctly)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

from flask import Flask, request, jsonify, Response, stream_with_context

from modules.logger import app_logger
from modules.config_manager import ConfigManager
from modules.router_factory import create_router
from modules.onnx_runner import SpecificModelRunner
from modules.ai_engine import AIEngine
from modules.expert_runner import ExpertDispatcher
from modules.plugin_manager import PluginManager

# ---------------------------------------------------------------------------
# Rate Limiter (sliding window per IP)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple sliding-window rate limiter per IP address."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # seconds

    def _cleanup_stale(self):
        """Remove IPs with no recent activity to prevent unbounded memory growth."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_ips = [
            ip for ip, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] < now - self._window * 2
        ]
        for ip in stale_ips:
            del self._requests[ip]

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            self._cleanup_stale()
            if client_ip not in self._requests:
                self._requests[client_ip] = deque()
            timestamps = self._requests[client_ip]
            while timestamps and timestamps[0] < now - self._window:
                timestamps.popleft()
            if len(timestamps) >= self._max_requests:
                return False
            timestamps.append(now)
            return True

_rate_limiter = _RateLimiter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_VERSION = "0.4.0"
DEFAULT_MODEL  = "l3mcore"


def _load_available_models(config_manager) -> list:
    """
    Builds the list of models announced in /v1/models and /api/tags.
    Generic mode: reads config/experts.json.
    Model mode: uses static list of known labels.
    """
    import json as _json
    cfg = config_manager.get('router', {})
    mode = cfg.get('mode', 'generic').lower()

    if mode == 'generic':
        cats_file = cfg.get('categories_file', 'config/experts.json')
        models = [DEFAULT_MODEL]
        if os.path.exists(cats_file):
            with open(cats_file, encoding='utf-8') as f:
                data = _json.load(f)
            from modules.config_manager import deobfuscate_value
            data = deobfuscate_value(data)
            experts = data.get('experts', [])
            for entry in experts:
                    label = entry.get('label', '').strip()
                    if label:
                        models.append(label)
        return models
    else:
        return [DEFAULT_MODEL, "malbec", "syrah", "pinot", "chardonnay", "grape-route"]


def _validate_api_keys(config_manager) -> None:
    """Logs warnings for API-type experts missing their configured API keys."""
    try:
        cats_file = config_manager.get('router', {}).get('categories_file', 'config/experts.json')
        if not os.path.exists(cats_file):
            return
        with open(cats_file, encoding='utf-8') as f:
            data = json.load(f)
        from modules.config_manager import deobfuscate_value
        data = deobfuscate_value(data)
        for expert in data.get('experts', []):
            if expert.get('type', '').lower() == 'api':
                env_var = expert.get('api_key_env', '')
                if env_var and not os.environ.get(env_var):
                    label = expert.get('label', 'unknown')
                    app_logger.warning(
                        f"API expert '{label}': environment variable '{env_var}' is not set. "
                        f"Inference will fail until the key is configured."
                    )
    except Exception as e:
        app_logger.debug(f"_validate_api_keys: {e}")


# ---------------------------------------------------------------------------
# Core MoE Initialization (singleton shared across requests)
# ---------------------------------------------------------------------------

class _Core:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._init()
        return cls._instance

    @classmethod
    def reload_experts(cls):
        instance = cls.get()
        with cls._lock:
            app_logger.info("l3mcore Core: Recargando configuraciones de expertos en el aire...")
            try:
                # 1. Recargar GenericRouter
                router = instance.get("router")
                if hasattr(router, "reload_categories"):
                    router.reload_categories()
                
                # 2. Recargar modelos disponibles en api_server
                config = instance.get("config")
                available = _load_available_models(config)
                
                instance["available_models"] = available
                instance["expert_models"] = [m for m in available if m != DEFAULT_MODEL]
                
                app_logger.info(f"l3mcore Core: Recarga automática finalizada. Modelos enrutables: {available}")
                instance["plugin_mgr"].hook_on_startup(instance)
            except Exception as e:
                app_logger.error(f"l3mcore Core: Error durante la recarga en caliente: {e}")

    @staticmethod
    def _init():
        app_logger.info("Initializing l3mcore Core...")
        config = ConfigManager()
        router = create_router(config)
        runner = SpecificModelRunner(
            models_base_path="models",
            stats_path="data/model_stats.json"
        )
        ai_engine = AIEngine(config_manager=config)
        dispatcher = ExpertDispatcher(runner, ai_engine, config_manager=config)
        plugin_mgr = PluginManager()

        _validate_api_keys(config)

        available = _load_available_models(config)
        expert_models = [m for m in available if m != DEFAULT_MODEL]
        app_logger.info(f"l3mcore Core ready. Models: {available}")
        return {
            "config": config,
            "router": router,
            "runner": runner,
            "ai_engine": ai_engine,
            "dispatcher": dispatcher,
            "plugin_mgr": plugin_mgr,
            "available_models": available,
            "expert_models": expert_models,
        }


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def _extract_routing_context(messages: list, max_messages: int = 3,
                              max_chars: int = 1600) -> dict:
    """
    Extracts routing context from the conversation history.

    Returns a dict with:
      - last_user_text:  text of the most recent user message (Step 1).
      - context_text:    concatenation of the last N user messages (Step 2).
    """
    if not messages or not isinstance(messages, list):
        return {"last_user_text": "", "context_text": ""}

    user_messages = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content)
        text = text.strip()
        if text:
            user_messages.append(text)

    last_user_text = user_messages[-1] if user_messages else ""

    recent = user_messages[-max_messages:] if len(user_messages) > 1 else []
    context_text = " ".join(recent)
    if len(context_text) > max_chars:
        context_text = context_text[-max_chars:]

    if not context_text:
        context_text = last_user_text
    if len(context_text) > max_chars:
        context_text = context_text[-max_chars:]

    return {
        "last_user_text": last_user_text,
        "context_text": context_text,
    }



def _notify_transparency(label: str, score: float) -> None:
    """
    Pushes the resolved label+score into the routing_transparency plugin
    if it is loaded, so the footer can display the confidence value.
    This is a best-effort call; any failure is silently ignored.
    """
    try:
        import sys
        plugin_module = sys.modules.get("l3mcore_plugin.routing_transparency")
        if plugin_module and hasattr(plugin_module, "set_route_score"):
            plugin_module.set_route_score(label, score)
    except Exception:
        pass


def _fire_failure_webhook(failed_label: str, reason: str) -> None:
    """
    Delegates failure notification to plugins via hook_on_expert_failure.
    """
    try:
        core = _Core.get()
        core["plugin_mgr"].hook_on_expert_failure(failed_label, reason)
    except Exception as exc:
        app_logger.warning(f"[Webhook] Failed to dispatch failure hook for '{failed_label}': {exc}")


def _clean_assistant_response(text: str) -> str:
    """
    Cleans up the assistant's final response to remove raw JSON tool calls
    (e.g., {"name": "...", "parameters": {...}}).
    """
    if not text:
        return text

    import json
    i = 0
    n = len(text)
    ranges_to_remove = []
    while i < n:
        if text[i] == '{':
            # Find the matching closing brace, tracking nested depth and strings
            depth = 1
            j = i + 1
            in_string = False
            escaped = False
            while j < n and depth > 0:
                char = text[j]
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                j += 1
            if depth == 0:
                # We found a candidate JSON block from i to j
                candidate = text[i:j]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "name" in obj and ("parameters" in obj or "arguments" in obj or "parameter" in obj):
                        # It is a tool call JSON block! Mark it for removal
                        ranges_to_remove.append((i, j))
                        i = j - 1
                except Exception:
                    pass
        i += 1

    # Remove the ranges from back to front
    new_text = text
    for start, end in reversed(ranges_to_remove):
        # Clean any surrounding whitespace/newlines as well
        prefix = new_text[:start]
        suffix = new_text[end:]
        prefix = prefix.rstrip()
        suffix = suffix.lstrip()
        # Keep a clean spacing between paragraphs if there was text before/after
        if prefix and suffix:
            new_text = prefix + "\n\n" + suffix
        else:
            new_text = prefix + suffix

    return new_text.strip()


def _run_inference(messages: list, model_hint: str) -> tuple[str, str]:
    """
    Wrapper for _run_inference_impl that checks security interceptor
    and records local telemetry.
    """
    core = _Core.get()
    config = core["config"]
    router_cfg = config.get('router', {})
    ctx_messages = router_cfg.get('context_messages', 3)
    ctx_chars = router_cfg.get('context_max_chars', 1600)
    
    routing_ctx = _extract_routing_context(messages, ctx_messages, ctx_chars)
    plugin_mgr = core["plugin_mgr"]
    last_text = plugin_mgr.hook_before_routing(routing_ctx["last_user_text"])

    # 1. Check security interceptor
    try:
        from modules.utils_text import sanitize
        intercepted = sanitize(last_text)
        if intercepted is not None:
            return intercepted, "canary_interceptor"
    except Exception as e:
        app_logger.warning(f"Security interceptor failed: {e}")

    t0 = time.monotonic()

    # 2. Run core inference
    res_text, used_lbl = _run_inference_impl(messages, model_hint)
    res_text = _clean_assistant_response(res_text)

    # 3. Record telemetry
    duration = time.monotonic() - t0
    try:
        from modules.session_store import push_context
        m_type = "unknown"
        if used_lbl == "fallback":
            m_type = "local-gguf"
        elif used_lbl == "error":
            m_type = "error"
        else:
            try:
                if hasattr(core["router"], 'get_expert_config'):
                    cfg = core["router"].get_expert_config(used_lbl)
                    if cfg:
                        m_type = cfg.get("type", "local")
                        if m_type == "local":
                            m_type = f"local-{cfg.get('format', 'onnx')}"
            except Exception:
                pass
        push_context(used_lbl, m_type, last_text, res_text, duration)
    except Exception as te:
        app_logger.warning(f"Telemetry tracking failed: {te}")

    return res_text, used_lbl


def _run_inference_impl(messages: list, model_hint: str) -> tuple[str, str]:
    """
    Executes inference with cascading contextual routing.
    Returns: (response_text, used_model)
    """
    core = _Core.get()
    router        = core["router"]
    config        = core["config"]
    dispatcher    = core["dispatcher"]
    ai_engine     = core["ai_engine"]
    expert_models = core["expert_models"]
    plugin_mgr    = core["plugin_mgr"]

    router_cfg   = config.get('router', {})
    ctx_messages = router_cfg.get('context_messages', 3)
    ctx_chars    = router_cfg.get('context_max_chars', 1600)
    threshold    = router_cfg.get('confidence_threshold', 0.4)

    routing_ctx = _extract_routing_context(messages, ctx_messages, ctx_chars)

    tc_cfg = config.get('tool_calling', {})
    tc_enabled = tc_cfg.get('enabled', False)
    override_label = None
    if not tc_enabled:
        override_label = plugin_mgr.hook_override_route(messages)

    last_text = plugin_mgr.hook_before_routing(routing_ctx["last_user_text"])
    # Context text is left available for plugins like contextual_cascade
    ctx_text  = routing_ctx["context_text"]

    def _execute_expert(label: str, score: float = 0.0) -> str:
        # Notify the routing_transparency plugin of the selected label+score
        # before running the expert so the footer is always accurate.
        _notify_transparency(label, score)
        if hasattr(router, 'get_expert_config'):
            expert_config = router.get_expert_config(label)
            if expert_config:
                plugin_mgr.hook_before_expert(messages, expert_config)
                result = dispatcher.run(messages, expert_config)
                return plugin_mgr.hook_after_generation(result, label)
        cfg = {"type": "local", "format": "onnx", "label": label}
        plugin_mgr.hook_before_expert(messages, cfg)
        result = dispatcher.run(messages, cfg)
        return plugin_mgr.hook_after_generation(result, label)

    def _do_fallback() -> tuple[str, str]:
        try:
            # Si el historial contiene contexto de documentos, redirigimos al document-expert
            has_doc_context = False
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str) and "[CONTEXTO DE DOCUMENTOS DE PAPERLESS-NGX]" in content:
                        has_doc_context = True
                        break

            fallback_label = "document-expert" if has_doc_context else "fallback"
            app_logger.info(f"[Fallback] Usando enrutamiento de respaldo a: '{fallback_label}' (tiene contexto activo: {has_doc_context})")
            result = _execute_expert(fallback_label, score=0.0)
            return result, fallback_label
        except Exception as e:
            app_logger.error(f"Critical error in fallback engine: {e}")
            return "An internal error occurred. Please try again later.", "error"

    # ---------------------------------------------------------------------------
    # Phase 0: Tool Calling mode (si tool_calling.enabled = true en config.json)
    # ---------------------------------------------------------------------------
    tc_cfg = config.get('tool_calling', {})
    if tc_cfg.get('enabled', False):
        tc_label      = tc_cfg.get('expert_label', 'document-expert')
        tc_max_iter   = int(tc_cfg.get('max_iterations', 5))
        expert_config = router.get_expert_config(tc_label) if hasattr(router, 'get_expert_config') else None

        if expert_config:
            tools    = plugin_mgr.hook_get_tools()
            tc_msgs  = list(messages)  # copia para no mutar el historial original

            app_logger.info(
                f"[ToolCalling] Modo activo. Experto: '{tc_label}', "
                f"tools disponibles: {[t['function']['name'] for t in tools if isinstance(t, dict) and 'function' in t]}"
            )

            for iteration in range(tc_max_iter):
                try:
                    plugin_mgr.hook_before_expert(tc_msgs, expert_config)
                    result = dispatcher.run(tc_msgs, expert_config, tools=tools if tools else None)
                except Exception as e:
                    app_logger.error(f"[ToolCalling] Error en iteracion {iteration+1}: {e}")
                    break

                # El modelo respondio con texto normal -> fin del bucle
                if isinstance(result, str):
                    final = plugin_mgr.hook_after_generation(result, tc_label)
                    return final, tc_label

                # El modelo respondio con tool_calls -> ejecutar y re-llamar
                if isinstance(result, dict) and result.get('tool_calls'):
                    raw_calls = result['tool_calls']

                    # Añadir el mensaje del asistente con los tool_calls al historial
                    tc_msgs.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": raw_calls
                    })

                    # Ejecutar cada herramienta y añadir su resultado
                    for call in raw_calls:
                        # Ollama y OpenAI tienen estructuras ligeramente distintas
                        if isinstance(call, dict):
                            call_id   = call.get('id', f'call_{iteration}')
                            fn        = call.get('function', {})
                            tool_name = fn.get('name', '')
                            args_raw  = fn.get('arguments', '{}')
                        else:
                            # Objeto litellm con atributos
                            call_id   = getattr(call, 'id', f'call_{iteration}')
                            fn        = getattr(call, 'function', None)
                            tool_name = getattr(fn, 'name', '') if fn else ''
                            args_raw  = getattr(fn, 'arguments', '{}') if fn else '{}'

                        try:
                            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except (json.JSONDecodeError, TypeError):
                            arguments = {}

                        app_logger.info(f"[ToolCalling] Ejecutando tool '{tool_name}' con args: {arguments}")
                        tool_result = plugin_mgr.hook_execute_tool(tool_name, arguments)
                        app_logger.info(f"[ToolCalling] Tool '{tool_name}' resultado: {str(tool_result)[:200]}")

                        tc_msgs.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result
                        })
                    continue

                # Respuesta inesperada
                app_logger.warning(f"[ToolCalling] Respuesta inesperada en iteracion {iteration+1}: {type(result)}")
                break

            app_logger.warning("[ToolCalling] Se alcanzo max_iterations o fallo el bucle. Usando fallback.")
            return _do_fallback()
        else:
            app_logger.warning(
                f"[ToolCalling] Experto '{tc_label}' no encontrado en experts.json. "
                "Desactivando tool calling para esta peticion."
            )

    # Phase 0b: Plugin forced route (comportamiento clasico)
    if override_label:
        try:
            app_logger.info(f"Plugin forced route to: {override_label}")
            result = _execute_expert(override_label, score=1.0)
            return result, override_label
        except Exception as e:
            app_logger.error(f"[Auto-Correction] Plugin forced route '{override_label}' failed: {e}. Redirecting to fallback.")
            _fire_failure_webhook(override_label, str(e))
            return _do_fallback()

    # Phase 1.5: Regex triggers matching
    if last_text:
        try:
            if hasattr(router, 'categories') and router.categories:
                for label, cat_data in router.categories.items():
                    cfg = cat_data.get('config', {})
                    triggers = cfg.get("regex_triggers", [])
                    if not isinstance(triggers, list):
                        continue
                    for pattern in triggers:
                        if isinstance(pattern, str) and pattern:
                            if re.search(pattern, last_text, re.IGNORECASE):
                                app_logger.info(f"[Regex] Matched '{pattern}' -> routing directly to '{label}'")
                                result = _execute_expert(label, score=1.0)
                                return result, label
        except Exception as e:
            app_logger.error(f"[Regex] Error evaluating triggers: {e}")

    # Phase 2: Cascade Step 1 - evaluate last user message only
    label, score = router.predict(last_text)
    if label and label not in ('null', 'fallback') and score >= threshold:
        app_logger.info(f"[Cascade] Step 1: '{last_text[:60]}' -> {label} ({score:.2f})")
        try:
            result = _execute_expert(label, score=score)
            return result, label
        except Exception as e:
            app_logger.error(f"[Auto-Correction] Routed expert '{label}' failed: {e}. Redirecting to fallback.")
            _fire_failure_webhook(label, str(e))
            return _do_fallback()
    else:
        app_logger.info(f"[Cascade] Step 1: score {score:.2f} below threshold ({threshold}). Escalating to Step 2.")
        
        # Phase 3: Cascade Step 2 - evaluate context via plugin
        target = plugin_mgr.hook_on_router_low_confidence(messages, label, score)
        if target:
            app_logger.info(f"[Cascade] Step 2 plugin overrode route to '{target}'")
            try:
                result = _execute_expert(target, score=1.0)
                return result, target
            except Exception as e:
                app_logger.error(f"[Auto-Correction] Routed expert '{target}' failed: {e}. Redirecting to fallback.")
                _fire_failure_webhook(target, str(e))
                return _do_fallback()

    # Phase 4: Fallback
    app_logger.info("[Cascade] No expert matched. Using fallback.")
    return _do_fallback()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB cap


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-RateLimit-Limit"] = str(_rate_limiter._max_requests)
    response.headers["X-RateLimit-Window"] = str(_rate_limiter._window)

    try:
        core = _Core.get()
        cors_cfg = core["config"].get("cors", {})
        if cors_cfg.get("enabled", False):
            origin = cors_cfg.get("origin", "*")
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Max-Age"] = "86400"
    except Exception:
        pass

    return response



@app.before_request
def call_plugin_before_request():
    client_ip = request.remote_addr or "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        return jsonify({
            "error": {
                "message": "Rate limit exceeded. Please slow down.",
                "type": "rate_limit_error"
            }
        }), 429

    core = _Core.get()
    if core and "plugin_mgr" in core:
        return core["plugin_mgr"].hook_before_request(request)
    return None


_NON_PRINTABLE = re.compile(r'[\x00-\x1f\x7f]')


def _safe_log(text: str, max_len: int = 120) -> str:
    """Strip control characters and truncate before writing to logs."""
    cleaned = _NON_PRINTABLE.sub(' ', text)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


# -- OpenAI format helpers --------------------------------------------------

def _openai_model_object(name: str) -> dict:
    return {
        "id": name,
        "object": "model",
        "created": 1700000000,
        "owned_by": "l3mcore",
    }


def _openai_chat_chunk(content: str, model: str, finish_reason=None) -> str:
    """Creates an SSE chunk in OpenAI streaming format."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _openai_chat_response(content: str, model: str) -> dict:
    """Full response (no streaming) in OpenAI format."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# -- OpenAI endpoints -------------------------------------------------------

@app.route("/v1/models", methods=["GET"])
def list_models_openai():
    available = _Core.get()["available_models"]
    return jsonify({
        "object": "list",
        "data": [_openai_model_object(m) for m in available],
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    content_type = request.content_type or ""
    if "application/json" not in content_type and "multipart/form-data" not in content_type:
        return jsonify({
            "error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}
        }), 415

    body = request.get_json(force=True, silent=True) or {}
    messages    = body.get("messages") or []
    model_hint  = body.get("model", DEFAULT_MODEL)
    do_stream   = body.get("stream", False)

    routing_ctx = _extract_routing_context(messages)
    user_text = routing_ctx["last_user_text"]
    if not user_text:
        return jsonify({"error": {"message": "No user message found", "type": "invalid_request_error"}}), 400

    app_logger.info(f"[/v1/chat] model={model_hint!r} stream={do_stream} text={_safe_log(user_text)!r}")

    if do_stream:
        def generate():
            try:
                response_text, used_model = _run_inference(messages, model_hint)
                # Emit content in a single chunk (ONNX experts don't do real streaming)
                yield _openai_chat_chunk(response_text, used_model)
                yield _openai_chat_chunk("", used_model, finish_reason="stop")
                yield "data: [DONE]\n\n"
            except Exception as e:
                app_logger.error(f"Streaming error: {e}")
                err = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        try:
            response_text, used_model = _run_inference(messages, model_hint)
            return jsonify(_openai_chat_response(response_text, used_model))
        except Exception as e:
            app_logger.error(f"Inference error: {e}")
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


# -- Ollama-compatible endpoints --------------------------------------------

@app.route("/api/version", methods=["GET"])
def ollama_version():
    return jsonify({"version": SERVER_VERSION})


@app.route("/api/tags", methods=["GET"])
def ollama_tags():
    """Ollama /api/tags - lists models in Ollama format."""
    available = _Core.get()["available_models"]
    expert_set = set(available) - {DEFAULT_MODEL}
    models = []
    for name in available:
        models.append({
            "name": name,
            "model": name,
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "onnx" if name in expert_set else "mixed",
                "family": "l3mcore",
                "families": ["l3mcore"],
                "parameter_size": "unknown",
                "quantization_level": "Q4",
            }
        })
    return jsonify({"models": models})


@app.route("/api/chat", methods=["POST"])
def ollama_chat():
    """
    Ollama POST /api/chat
    Body: { model, messages: [{role, content}], stream }
    """
    content_type = request.content_type or ""
    if "application/json" not in content_type:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    body = request.get_json(force=True, silent=True) or {}
    messages   = body.get("messages") or []
    model_hint = body.get("model", DEFAULT_MODEL)
    do_stream  = body.get("stream", True)  # Ollama defaults to stream=true

    routing_ctx = _extract_routing_context(messages)
    user_text = routing_ctx["last_user_text"]
    if not user_text:
        return jsonify({"error": "No user message found"}), 400

    app_logger.info(f"[/api/chat] model={model_hint!r} stream={do_stream} text={_safe_log(user_text)!r}")

    def _ollama_chunk(content: str, model: str, done: bool) -> str:
        obj = {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": content},
            "done": done,
        }
        if done:
            obj.update({
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "eval_duration": 0,
            })
        return json.dumps(obj) + "\n"

    if do_stream:
        def generate():
            try:
                response_text, used_model = _run_inference(messages, model_hint)
                yield _ollama_chunk(response_text, used_model, done=False)
                yield _ollama_chunk("", used_model, done=True)
            except Exception as e:
                app_logger.error(f"Error in /api/chat streaming: {e}")
                yield json.dumps({"error": str(e)}) + "\n"

        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            response_text, used_model = _run_inference(messages, model_hint)
            return Response(
                _ollama_chunk(response_text, used_model, done=True),
                mimetype="application/json",
            )
        except Exception as e:
            app_logger.error(f"Error in /api/chat: {e}")
            return jsonify({"error": str(e)}), 500

# -- Routing diagnostic endpoint -------------------------------------------

_ROUTE_TEXT_MAX = 2000  # characters accepted in the query parameter
_ROUTE_TEXT_RE  = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')  # strip control chars


@app.route("/v1/route", methods=["GET", "POST"])
def route_inspect():
    """
    Diagnostic endpoint: runs the router against a text and returns the
    full scoring breakdown without generating any model response.

    GET  /v1/route?text=<your+prompt>
    POST /v1/route  body: {"text": "your prompt"}

    Response:
      {
        "expert":       "programador",
        "score":        0.87,
        "method":       "embedding",
        "cascade_step": 1,
        "top_experts":  [{"expert": "...", "score": ...}, ...]
      }
    """
    # --- Extract input text -------------------------------------------------
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        raw_text = body.get("text", "")
    else:
        raw_text = request.args.get("text", "")

    if not isinstance(raw_text, str):
        return jsonify({"error": {"message": "'text' must be a string", "type": "invalid_request_error"}}), 400

    # Strip control characters and cap length
    text = _ROUTE_TEXT_RE.sub(' ', raw_text).strip()[:_ROUTE_TEXT_MAX]

    if not text:
        return jsonify({"error": {"message": "'text' is required and cannot be empty", "type": "invalid_request_error"}}), 400

    app_logger.info(f"[/v1/route] text={_safe_log(text)!r}")

    # --- Run router ---------------------------------------------------------
    core   = _Core.get()
    router = core["router"]
    config = core["config"]

    router_cfg = config.get('router', {})
    threshold  = router_cfg.get('confidence_threshold', 0.4)

    result: dict = {
        "expert":       "fallback",
        "score":        0.0,
        "method":       "fallback",
        "cascade_step": None,
        "top_experts":  [],
    }

    try:
        # Step 0: Regex triggers matching
        matched_label = None
        if hasattr(router, 'categories') and router.categories:
            for label, cat_data in router.categories.items():
                cfg = cat_data.get('config', {})
                triggers = cfg.get("regex_triggers", [])
                if not isinstance(triggers, list):
                    continue
                for pattern in triggers:
                    if isinstance(pattern, str) and pattern:
                        if re.search(pattern, text, re.IGNORECASE):
                            matched_label = label
                            break
                if matched_label:
                    break

        if matched_label:
            result["expert"]       = matched_label
            result["score"]        = 1.0
            result["method"]       = "regex"
            result["cascade_step"] = 1
        else:
            # Step 1: direct prediction
            label, score = router.predict(text)
            if label and label not in ('null', 'fallback') and score >= threshold:
                result["expert"]       = label
                result["score"]        = round(score, 4)
                result["method"]       = getattr(router, 'router_type', 'unknown')
                result["cascade_step"] = 1
            else:
                result["method"] = "fallback"

        # Build top_experts from category_embeddings if available
        # (embedding mode only — gives the full breakdown to the caller)
        cat_emb = getattr(router, 'category_embeddings', {})
        if cat_emb:
            import math
            sw  = getattr(router, 'scoring_weights', {})
            tmp = getattr(router, 'softmax_temperature', 0.15)
            from modules.utils_router import clean_text
            from sentence_transformers import util as st_util
            clean = clean_text(text)
            if clean and router._model is not None:
                query_vec = router._model.encode(
                    "query: " + clean, convert_to_tensor=True, show_progress_bar=False
                )
                raw: dict[str, float] = {
                    lbl: router._embed_score(query_vec, data)
                    for lbl, data in cat_emb.items()
                }
                max_raw = max(raw.values()) if raw else 0.0
                exp_s   = {l: math.exp((s - max_raw) / tmp) for l, s in raw.items()}
                total   = sum(exp_s.values()) or 1.0
                norm    = {l: v / total for l, v in exp_s.items()}
                top = sorted(norm.items(), key=lambda x: -x[1])[:5]
                result["top_experts"] = [
                    {"expert": lbl, "score": round(sc, 4)} for lbl, sc in top
                ]

    except Exception as e:
        app_logger.error(f"[/v1/route] Router error: {e}")
        result["error"] = "router_error"

    return jsonify(result)


# -- Ollama model discovery --------------------------------------------------

@app.route("/v1/discover", methods=["GET"])
def discover_ollama():
    """
    Queries the local Ollama instance and returns models that are not yet
    configured as experts, along with a ready-to-paste experts.json snippet.

    Query params:
      url  — Ollama base URL (default: http://127.0.0.1:11434)
    """
    import urllib.request as _ur

    raw_url = request.args.get("url", "http://127.0.0.1:11434").strip().rstrip("/")

    # Scheme guard
    if not raw_url.startswith(("http://", "https://")):
        return jsonify({"error": {"message": "url must start with http:// or https://", "type": "invalid_request_error"}}), 400

    # SSRF protection: validate URL is not targeting internal/private networks
    try:
        from urllib.parse import urlparse as _urlparse
        parsed = _urlparse(raw_url)
        import ipaddress as _ipaddress
        import socket
        hostname = parsed.hostname or ""
        # Block cloud metadata and link-local
        try:
            resolved = socket.getaddrinfo(hostname, None)
            for family, _, _, _, sockaddr in resolved:
                addr = _ipaddress.ip_address(sockaddr[0])
                if any(addr in net for net in [
                    _ipaddress.ip_network("169.254.0.0/16"),
                    _ipaddress.ip_network("100.64.0.0/10"),
                    _ipaddress.ip_network("fd00::/8"),
                ]):
                    return jsonify({"error": {"message": "URL targets a blocked internal network", "type": "invalid_request_error"}}), 400
        except (socket.gaierror, ValueError):
            pass
    except Exception:
        pass

    core            = _Core.get()
    configured      = set(core["expert_models"])
    suggestions     = []
    raw_models      = []
    error_msg       = None

    try:
        req = _ur.Request(f"{raw_url}/api/tags", headers={"Accept": "application/json"})
        with _ur.urlopen(req, timeout=5) as resp:
            data       = json.loads(resp.read().decode("utf-8"))
            raw_models = data.get("models", [])
    except Exception as exc:
        error_msg = str(exc)

    for entry in raw_models:
        name = entry.get("name", "") or entry.get("model", "")
        if not name:
            continue
        label = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.split(":")[0])[:48]
        if label in configured:
            continue
        suggestions.append({
            "model_name": name,
            "suggested_label": label,
            "snippet": {
                "label":       label,
                "description": f"Expert using {name}. Add keywords and a description.",
                "keywords":    [],
                "type":        "ollama",
                "url":         raw_url,
                "model_name":  name,
            },
        })

    result: dict = {
        "ollama_url":         raw_url,
        "configured_experts": sorted(configured),
        "unconfigured_models": suggestions,
    }
    if error_msg:
        result["error"] = error_msg
        app_logger.warning(f"[/v1/discover] Could not reach Ollama at {raw_url}: {error_msg}")

    return jsonify(result)


# -- Additional compatibility routes ----------------------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "l3mcore",
        "version": SERVER_VERSION,
        "description": "Light Easy Mix Of Experts – OpenAI & Ollama compatible API",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/route", "/v1/discover", "/api/tags", "/api/chat", "/api/version", "/health"],
    })



@app.route("/health", methods=["GET"])
def health():
    """Returns the operational status of every core component."""
    core = _Core.get()
    config = core["config"]

    health_cfg = config.get("health", {})
    if health_cfg.get("auth_required", False):
        auth_header = request.headers.get("Authorization", "")
        expected_token = health_cfg.get("auth_token", "")
        if expected_token and auth_header != f"Bearer {expected_token}":
            return jsonify({"error": {"message": "Unauthorized", "type": "auth_error"}}), 401

    router = core["router"]
    runner = core["runner"]
    ai_engine = core["ai_engine"]
    plugin_mgr = core["plugin_mgr"]

    router_status = "ok"
    router_mode = getattr(router, 'router_type', 'model')
    router_enabled = getattr(router, 'enabled', False)
    if not router_enabled:
        router_status = "degraded (keyword fallback only)"

    plugins_loaded = len(getattr(plugin_mgr, '_plugins', []))

    ai_ready = getattr(ai_engine, 'is_ready', False)
    models_in_memory = list(getattr(runner, 'sessions', {}).keys())

    # Check if config.json has been modified since startup
    config.check_for_changes()

    status = {
        "status": "ok",
        "version": SERVER_VERSION,
        "router": {
            "mode": router_mode,
            "status": router_status,
            "cache_size": len(getattr(router, '_predict_cache', {})),
        },
        "onnx_runner": {
            "models_in_memory": models_in_memory,
            "max_models": getattr(runner, 'max_models', 3),
        },
        "ai_engine": {
            "model": getattr(ai_engine, 'model_path', 'unknown'),
            "loaded": ai_ready,
        },
        "plugins": {
            "loaded": plugins_loaded,
            "names": [
                getattr(p, '__name__', '?').replace('l3mcore_plugin.', '')
                for p in getattr(plugin_mgr, '_plugins', [])
            ],
        },
        "available_models": core["available_models"],
    }
    return jsonify(status)


@app.route("/v1/health/experts", methods=["GET"])
@app.route("/health/experts", methods=["GET"])
def health_experts():
    """
    Checks the connectivity and status of all configured experts.
    Returns a JSON report showing which backends are active/reachable.
    """
    core = _Core.get()
    router_cfg_file = core["config"].get("router", {}).get("categories_file", "config/experts.json")
    
    try:
        with open(router_cfg_file, encoding="utf-8") as f:
            data = json.load(f)
        from modules.config_manager import deobfuscate_value
        data = deobfuscate_value(data)
        experts_list = data.get("experts", [])
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not read experts file: {e}"}), 500

    results = {}
    
    import urllib.request as _ur
    import urllib.error as _ue
    import os
    
    for exp in experts_list:
        label = exp.get("label")
        if not label:
            continue
            
        expert_type = exp.get("type", "local").lower()
        status = "unknown"
        details = ""
        
        if expert_type == "ollama":
            url = exp.get("url", "http://127.0.0.1:11434").rstrip("/")
            try:
                req = _ur.Request(f"{url}/api/tags")
                with _ur.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        status = "reachable"
                        details = f"Ollama responds on {url}"
                    else:
                        status = "unreachable"
                        details = f"HTTP status {resp.status}"
            except Exception as e:
                status = "unreachable"
                details = str(e)
                
        elif expert_type == "api":
            env_var = exp.get("api_key_env", "")
            if env_var:
                if os.environ.get(env_var):
                    status = "configured"
                    details = f"API key set in environment ({env_var})"
                else:
                    status = "missing_key"
                    details = f"Environment variable '{env_var}' is not set"
            else:
                status = "configured"
                details = "No specific API key environment variable required"
                
        elif expert_type == "local":
            model_format = exp.get("format", "onnx").lower()
            model_path = exp.get("model_path", "")
            if model_format == "onnx":
                if model_path and os.path.exists(model_path):
                    status = "ready"
                    details = f"Local ONNX model path exists ({model_path})"
                else:
                    status = "missing_model"
                    details = f"Model path not found ({model_path})"
            elif model_format == "gguf":
                if model_path and os.path.exists(model_path):
                    status = "ready"
                    details = f"Local GGUF model path exists ({model_path})"
                else:
                    status = "missing_model"
                    details = f"Model path not found ({model_path})"
            else:
                status = "configured"
                details = f"Local expert with format: {model_format}"
                
        results[label] = {
            "type": expert_type,
            "status": status,
            "details": details
        }
        
    return jsonify({
        "status": "ok",
        "experts": results
    })


# Keyword enrichment has been moved to a plugin.


# ---------------------------------------------------------------------------
# WSGI entrypoint (Gunicorn / uWSGI) and dev entrypoint
# ---------------------------------------------------------------------------

def _start_experts_watcher():
    """Starts background thread to watch experts.json and config.json changes."""
    def watch():
        exp_path = "config/experts.json"
        cfg_path = "config/config.json"
        
        last_exp_mtime = os.path.getmtime(exp_path) if os.path.exists(exp_path) else 0.0
        last_cfg_mtime = os.path.getmtime(cfg_path) if os.path.exists(cfg_path) else 0.0
        
        while True:
            time.sleep(2)
            try:
                # Watch experts.json
                if os.path.exists(exp_path):
                    current_exp_mtime = os.path.getmtime(exp_path)
                    if current_exp_mtime > last_exp_mtime:
                        last_exp_mtime = current_exp_mtime
                        app_logger.info("Watcher: experts.json modification detected. Hot-reloading experts...")
                        _Core.reload_experts()

                # Watch config.json
                if os.path.exists(cfg_path):
                    current_cfg_mtime = os.path.getmtime(cfg_path)
                    if current_cfg_mtime > last_cfg_mtime:
                        last_cfg_mtime = current_cfg_mtime
                        app_logger.info("Watcher: config.json modification detected. Hot-reloading configuration...")
                        ConfigManager().load()
                        _Core.reload_experts()
            except Exception as e:
                app_logger.error(f"Watcher error: {e}")

    watcher_thread = threading.Thread(target=watch, daemon=True)
    watcher_thread.start()
    app_logger.info("Watcher: Started background thread for configuration and experts monitoring.")


def _print_startup_summary(core: dict) -> None:
    """Prints a summary of the loaded configuration at startup."""
    config = core["config"]
    router = core["router"]
    plugin_mgr = core["plugin_mgr"]
    available = core["available_models"]

    router_cfg = config.get('router', {})
    rl_cfg = config.get('rate_limiting', {})

    app_logger.info("=" * 50)
    app_logger.info("l3mcore - Startup Summary")
    app_logger.info("=" * 50)
    app_logger.info(f"  Router mode:     {router_cfg.get('mode', 'generic')}")
    app_logger.info(f"  Router type:     {router_cfg.get('router_type', 'embedding')}")
    app_logger.info(f"  Confidence:      {router_cfg.get('confidence_threshold', 0.4)}")
    app_logger.info(f"  Experts loaded:  {len(available) - 1}")
    app_logger.info(f"  Models:          {', '.join(available)}")
    plugins = [getattr(p, '__name__', '?').replace('l3mcore_plugin.', '') for p in getattr(plugin_mgr, '_plugins', [])]
    app_logger.info(f"  Plugins:         {len(plugins)} ({', '.join(plugins) if plugins else 'none'})")
    tools = [getattr(t, '__name__', '?').replace('l3mcore_tool.', '') for t in getattr(plugin_mgr, '_tools', [])]
    app_logger.info(f"  Tools:           {len(tools)} ({', '.join(tools) if tools else 'none'})")
    rl_status = "enabled" if rl_cfg.get('enabled', True) else "disabled"
    app_logger.info(f"  Rate limiting:   {rl_status} ({rl_cfg.get('max_requests', 60)}/{rl_cfg.get('window_seconds', 60)}s)")
    app_logger.info("=" * 50)


def _bootstrap():
    """Pre-load core + plugins once before the first request."""
    core = _Core.get()
    _start_experts_watcher()
    core["plugin_mgr"].hook_on_startup(core)
    _print_startup_summary(core)


# Gunicorn calls this module-level; bootstrap when the module is imported
_bootstrap()


def run(host: str = "0.0.0.0", port: int = 11435, debug: bool = False):
    """
    Dev-only entrypoint (Flask built-in server).
    In production use Gunicorn: gunicorn -w 1 -b 0.0.0.0:11435 api_server:app
    """
    app_logger.info(f"[DEV] l3mcore API listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="l3mcore API Server (dev mode)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=args.debug)
