"""
l3mcore API Server — versión light

OpenAI- y Ollama-compatible. Versión limpia sin plugins de telemetría,
sin tool calling, sin litellm, sin endpoints de descubrimiento.

Endpoints:
  GET  /                    -> Info del servidor
  GET  /health              -> Estado de los componentes
  GET  /v1/models           -> Lista de expertos (formato OpenAI)
  POST /v1/chat/completions -> Inferencia + streaming (formato OpenAI)
  GET  /v1/route            -> Diagnóstico de enrutamiento
  GET  /api/tags            -> Lista de modelos (formato Ollama)
  POST /api/chat            -> Inferencia + streaming (formato Ollama)
  GET  /api/version         -> Versión del servidor
"""

import json
import os
import re
import time
import uuid
import threading
from collections import deque

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
# Rate Limiter (sliding window por IP)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Sliding-window rate limiter por dirección IP."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300

    def _cleanup_stale(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale = [
            ip for ip, ts in self._requests.items()
            if not ts or ts[-1] < now - self._window * 2
        ]
        for ip in stale:
            del self._requests[ip]

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            self._cleanup_stale()
            if client_ip not in self._requests:
                self._requests[client_ip] = deque()
            ts = self._requests[client_ip]
            while ts and ts[0] < now - self._window:
                ts.popleft()
            if len(ts) >= self._max_requests:
                return False
            ts.append(now)
            return True


_rate_limiter = _RateLimiter()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SERVER_VERSION = "0.5.0-light"
DEFAULT_MODEL  = "l3mcore"


def _load_available_models(config_manager) -> list:
    cfg = config_manager.get('router', {})
    mode = cfg.get('mode', 'generic').lower()

    if mode == 'generic':
        cats_file = cfg.get('categories_file', 'config/experts.json')
        models = [DEFAULT_MODEL]
        if os.path.exists(cats_file):
            with open(cats_file, encoding='utf-8') as f:
                data = json.load(f)
            from modules.config_manager import deobfuscate_value
            data = deobfuscate_value(data)
            for entry in data.get('experts', []):
                label = entry.get('label', '').strip()
                if label:
                    models.append(label)
        return models
    else:
        return [DEFAULT_MODEL, "malbec", "syrah", "pinot", "chardonnay", "grape-route"]


# ---------------------------------------------------------------------------
# Singleton Core
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
            app_logger.info("Core: Hot-reloading expert config...")
            try:
                router = instance.get("router")
                if hasattr(router, "reload_categories"):
                    router.reload_categories()
                config = instance.get("config")
                available = _load_available_models(config)
                instance["available_models"] = available
                instance["expert_models"] = [m for m in available if m != DEFAULT_MODEL]
                app_logger.info(f"Core: Reload done. Models: {available}")
            except Exception as e:
                app_logger.error(f"Core: Hot-reload error: {e}")

    @staticmethod
    def _init():
        app_logger.info("Initializing l3mcore Core (light)...")
        config     = ConfigManager()
        router     = create_router(config)
        runner     = SpecificModelRunner(
            models_base_path="models",
            stats_path="data/model_stats.json"
        )
        ai_engine  = AIEngine(config_manager=config)
        dispatcher = ExpertDispatcher(runner, ai_engine, config_manager=config)
        plugin_mgr = PluginManager()

        available     = _load_available_models(config)
        expert_models = [m for m in available if m != DEFAULT_MODEL]
        app_logger.info(f"Core ready. Models: {available}")
        return {
            "config":           config,
            "router":           router,
            "runner":           runner,
            "ai_engine":        ai_engine,
            "dispatcher":       dispatcher,
            "plugin_mgr":       plugin_mgr,
            "available_models": available,
            "expert_models":    expert_models,
        }


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

_NON_PRINTABLE = re.compile(r'[\x00-\x1f\x7f]')
_ROUTE_TEXT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_ROUTE_TEXT_MAX = 2000


def _safe_log(text: str, max_len: int = 120) -> str:
    cleaned = _NON_PRINTABLE.sub(' ', text)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _extract_routing_context(messages: list, max_messages: int = 3,
                              max_chars: int = 1600) -> dict:
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

    return {"last_user_text": last_user_text, "context_text": context_text}


def _run_inference(messages: list, model_hint: str) -> tuple[str, str]:
    """Wrapper de seguridad sobre _run_inference_impl."""
    core = _Core.get()
    config = core["config"]
    router_cfg = config.get('router', {})
    ctx_messages = router_cfg.get('context_messages', 3)
    ctx_chars    = router_cfg.get('context_max_chars', 1600)

    routing_ctx = _extract_routing_context(messages, ctx_messages, ctx_chars)
    plugin_mgr  = core["plugin_mgr"]
    last_text   = plugin_mgr.hook_before_routing(routing_ctx["last_user_text"])

    # Interceptor de seguridad
    try:
        from modules.utils_text import sanitize
        intercepted = sanitize(last_text)
        if intercepted is not None:
            return intercepted, "canary_interceptor"
    except Exception as e:
        app_logger.warning(f"Security interceptor failed: {e}")

    return _run_inference_impl(messages, model_hint)


def _run_inference_impl(messages: list, model_hint: str) -> tuple[str, str]:
    """
    Lógica de enrutamiento en cascada:
      1. Plugin override (hook_override_route)
      2. Regex triggers
      3. Predicción ML del router
      4. Fallback GGUF
    """
    core          = _Core.get()
    router        = core["router"]
    config        = core["config"]
    dispatcher    = core["dispatcher"]
    ai_engine     = core["ai_engine"]
    plugin_mgr    = core["plugin_mgr"]

    router_cfg   = config.get('router', {})
    ctx_messages = router_cfg.get('context_messages', 3)
    ctx_chars    = router_cfg.get('context_max_chars', 1600)
    threshold    = router_cfg.get('confidence_threshold', 0.4)

    routing_ctx = _extract_routing_context(messages, ctx_messages, ctx_chars)
    last_text   = plugin_mgr.hook_before_routing(routing_ctx["last_user_text"])

    override_label = plugin_mgr.hook_override_route(messages)

    def _execute_expert(label: str, score: float = 0.0) -> str:
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
            app_logger.info("[Fallback] Usando motor GGUF.")
            result = ai_engine.generate_response(
                _extract_routing_context(messages)["last_user_text"]
            )
            return plugin_mgr.hook_after_generation(result, "fallback"), "fallback"
        except Exception as e:
            app_logger.error(f"Critical error in fallback: {e}")
            return "An internal error occurred. Please try again later.", "error"

    # Phase 0: Plugin forced override
    if override_label:
        try:
            app_logger.info(f"Plugin forced route to: {override_label}")
            result = _execute_expert(override_label, score=1.0)
            return result, override_label
        except Exception as e:
            app_logger.error(f"Plugin route '{override_label}' failed: {e}. Fallback.")
            return _do_fallback()

    # Phase 1: Regex triggers
    if last_text:
        try:
            if hasattr(router, 'categories') and router.categories:
                for label, cat_data in router.categories.items():
                    triggers = cat_data.get('config', {}).get("regex_triggers", [])
                    for pattern in triggers:
                        if isinstance(pattern, str) and pattern:
                            if re.search(pattern, last_text, re.IGNORECASE):
                                app_logger.info(f"[Regex] '{pattern}' -> '{label}'")
                                result = _execute_expert(label, score=1.0)
                                return result, label
        except Exception as e:
            app_logger.error(f"[Regex] Error: {e}")

    # Phase 2: Predicción ML
    label, score = router.predict(last_text)
    if label and label not in ('null', 'fallback') and score >= threshold:
        app_logger.info(f"[Router] '{last_text[:60]}' -> {label} ({score:.2f})")
        try:
            result = _execute_expert(label, score=score)
            return result, label
        except Exception as e:
            app_logger.error(f"Expert '{label}' failed: {e}. Fallback.")
            return _do_fallback()
    else:
        app_logger.info(f"[Router] Score {score:.2f} below threshold ({threshold}). Fallback.")

    # Phase 3: Fallback GGUF
    return _do_fallback()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB


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
def enforce_rate_limit():
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


# --- Helpers OpenAI ---------------------------------------------------------

def _openai_model_object(name: str) -> dict:
    return {"id": name, "object": "model", "created": 1700000000, "owned_by": "l3mcore"}


def _openai_chunk(content: str, model: str, finish_reason=None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _openai_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# --- Endpoints OpenAI -------------------------------------------------------

@app.route("/v1/models", methods=["GET"])
def list_models_openai():
    available = _Core.get()["available_models"]
    return jsonify({"object": "list", "data": [_openai_model_object(m) for m in available]})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    content_type = request.content_type or ""
    if "application/json" not in content_type:
        return jsonify({"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}}), 415

    body       = request.get_json(force=True, silent=True) or {}
    messages   = body.get("messages") or []
    model_hint = body.get("model", DEFAULT_MODEL)
    do_stream  = body.get("stream", False)

    routing_ctx = _extract_routing_context(messages)
    user_text   = routing_ctx["last_user_text"]
    if not user_text:
        return jsonify({"error": {"message": "No user message found", "type": "invalid_request_error"}}), 400

    app_logger.info(f"[/v1/chat] model={model_hint!r} stream={do_stream} text={_safe_log(user_text)!r}")

    if do_stream:
        def generate():
            try:
                response_text, used_model = _run_inference(messages, model_hint)
                yield _openai_chunk(response_text, used_model)
                yield _openai_chunk("", used_model, finish_reason="stop")
                yield "data: [DONE]\n\n"
            except Exception as e:
                app_logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            response_text, used_model = _run_inference(messages, model_hint)
            return jsonify(_openai_response(response_text, used_model))
        except Exception as e:
            app_logger.error(f"Inference error: {e}")
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


# --- Diagnóstico de enrutamiento -------------------------------------------

@app.route("/v1/route", methods=["GET", "POST"])
def route_inspect():
    """Ejecuta el router sobre un texto y devuelve el scoring sin generar respuesta."""
    if request.method == "POST":
        raw_text = (request.get_json(force=True, silent=True) or {}).get("text", "")
    else:
        raw_text = request.args.get("text", "")

    if not isinstance(raw_text, str):
        return jsonify({"error": {"message": "'text' must be a string"}}), 400

    text = _ROUTE_TEXT_RE.sub(' ', raw_text).strip()[:_ROUTE_TEXT_MAX]
    if not text:
        return jsonify({"error": {"message": "'text' is required"}}), 400

    core      = _Core.get()
    router    = core["router"]
    threshold = core["config"].get('router', {}).get('confidence_threshold', 0.4)

    result = {"expert": "fallback", "score": 0.0, "method": "fallback", "top_experts": []}

    try:
        matched_label = None
        if hasattr(router, 'categories') and router.categories:
            for label, cat_data in router.categories.items():
                triggers = cat_data.get('config', {}).get("regex_triggers", [])
                for pattern in triggers:
                    if isinstance(pattern, str) and re.search(pattern, text, re.IGNORECASE):
                        matched_label = label
                        break
                if matched_label:
                    break

        if matched_label:
            result.update({"expert": matched_label, "score": 1.0, "method": "regex"})
        else:
            label, score = router.predict(text)
            if label and label not in ('null', 'fallback') and score >= threshold:
                result.update({
                    "expert": label,
                    "score": round(score, 4),
                    "method": getattr(router, 'router_type', 'unknown'),
                })

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
                raw = {lbl: router._embed_score(query_vec, d) for lbl, d in cat_emb.items()}
                max_raw = max(raw.values()) if raw else 0.0
                exp_s   = {l: math.exp((s - max_raw) / tmp) for l, s in raw.items()}
                total   = sum(exp_s.values()) or 1.0
                norm    = {l: v / total for l, v in exp_s.items()}
                top     = sorted(norm.items(), key=lambda x: -x[1])[:5]
                result["top_experts"] = [{"expert": l, "score": round(sc, 4)} for l, sc in top]

    except Exception as e:
        app_logger.error(f"[/v1/route] Router error: {e}")
        result["error"] = "router_error"

    return jsonify(result)


# --- Endpoints Ollama -------------------------------------------------------

@app.route("/api/version", methods=["GET"])
def ollama_version():
    return jsonify({"version": SERVER_VERSION})


@app.route("/api/tags", methods=["GET"])
def ollama_tags():
    available   = _Core.get()["available_models"]
    expert_set  = set(available) - {DEFAULT_MODEL}
    models = [
        {
            "name": name, "model": name,
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0, "digest": "",
            "details": {
                "parent_model": "", "format": "onnx" if name in expert_set else "mixed",
                "family": "l3mcore", "families": ["l3mcore"],
                "parameter_size": "unknown", "quantization_level": "Q4",
            },
        }
        for name in available
    ]
    return jsonify({"models": models})


@app.route("/api/chat", methods=["POST"])
def ollama_chat():
    content_type = request.content_type or ""
    if "application/json" not in content_type:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    body       = request.get_json(force=True, silent=True) or {}
    messages   = body.get("messages") or []
    model_hint = body.get("model", DEFAULT_MODEL)
    do_stream  = body.get("stream", True)

    routing_ctx = _extract_routing_context(messages)
    user_text   = routing_ctx["last_user_text"]
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
            obj.update({"total_duration": 0, "load_duration": 0,
                        "prompt_eval_count": 0, "eval_count": 0, "eval_duration": 0})
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
            return Response(_ollama_chunk(response_text, used_model, done=True), mimetype="application/json")
        except Exception as e:
            app_logger.error(f"Error in /api/chat: {e}")
            return jsonify({"error": str(e)}), 500


# --- Endpoints generales ----------------------------------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "l3mcore",
        "version": SERVER_VERSION,
        "description": "Light Easy Mix Of Experts — OpenAI & Ollama compatible",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/route",
                      "/api/tags", "/api/chat", "/api/version", "/health"],
    })


@app.route("/health", methods=["GET"])
def health():
    core      = _Core.get()
    config    = core["config"]

    health_cfg = config.get("health", {})
    if health_cfg.get("auth_required", False):
        auth_header    = request.headers.get("Authorization", "")
        expected_token = health_cfg.get("auth_token", "")
        if expected_token and auth_header != f"Bearer {expected_token}":
            return jsonify({"error": {"message": "Unauthorized", "type": "auth_error"}}), 401

    router     = core["router"]
    runner     = core["runner"]
    ai_engine  = core["ai_engine"]
    plugin_mgr = core["plugin_mgr"]

    router_enabled = getattr(router, 'enabled', False)
    router_status  = "ok" if router_enabled else "degraded (keyword fallback only)"

    return jsonify({
        "status":    "ok",
        "version":   SERVER_VERSION,
        "router": {
            "mode":       getattr(router, 'router_type', 'model'),
            "status":     router_status,
            "cache_size": len(getattr(router, '_predict_cache', {})),
        },
        "onnx_runner": {
            "models_in_memory": list(getattr(runner, 'sessions', {}).keys()),
            "max_models":       getattr(runner, 'max_models', 3),
        },
        "ai_engine": {
            "model":  getattr(ai_engine, 'model_path', 'unknown'),
            "loaded": getattr(ai_engine, 'is_ready', False),
        },
        "plugins": {
            "loaded": len(getattr(plugin_mgr, '_plugins', [])),
        },
        "available_models": core["available_models"],
    })


# ---------------------------------------------------------------------------
# Watcher + Bootstrap
# ---------------------------------------------------------------------------

def _start_experts_watcher():
    def watch():
        exp_path = "config/experts.json"
        cfg_path = "config/config.json"
        last_exp = os.path.getmtime(exp_path) if os.path.exists(exp_path) else 0.0
        last_cfg = os.path.getmtime(cfg_path) if os.path.exists(cfg_path) else 0.0

        while True:
            time.sleep(2)
            try:
                if os.path.exists(exp_path):
                    mtime = os.path.getmtime(exp_path)
                    if mtime > last_exp:
                        last_exp = mtime
                        app_logger.info("Watcher: experts.json changed. Hot-reloading...")
                        _Core.reload_experts()
                if os.path.exists(cfg_path):
                    mtime = os.path.getmtime(cfg_path)
                    if mtime > last_cfg:
                        last_cfg = mtime
                        app_logger.info("Watcher: config.json changed. Hot-reloading...")
                        ConfigManager().load()
                        _Core.reload_experts()
            except Exception as e:
                app_logger.error(f"Watcher error: {e}")

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    app_logger.info("Watcher: started.")


def _print_startup_summary(core: dict) -> None:
    config     = core["config"]
    router     = core["router"]
    plugin_mgr = core["plugin_mgr"]
    available  = core["available_models"]
    router_cfg = config.get('router', {})

    app_logger.info("=" * 50)
    app_logger.info("l3mcore light — Startup Summary")
    app_logger.info("=" * 50)
    app_logger.info(f"  Version:     {SERVER_VERSION}")
    app_logger.info(f"  Router mode: {router_cfg.get('mode', 'generic')}")
    app_logger.info(f"  Router type: {router_cfg.get('router_type', 'embedding')}")
    app_logger.info(f"  Threshold:   {router_cfg.get('confidence_threshold', 0.4)}")
    app_logger.info(f"  Experts:     {len(available) - 1}")
    app_logger.info(f"  Models:      {', '.join(available)}")
    plugins = [getattr(p, '__name__', '?').replace('l3mcore_plugin.', '')
               for p in getattr(plugin_mgr, '_plugins', [])]
    app_logger.info(f"  Plugins:     {len(plugins)} ({', '.join(plugins) or 'none'})")
    app_logger.info("=" * 50)


def _bootstrap():
    """Pre-carga el Core antes de la primera petición."""
    core = _Core.get()
    _start_experts_watcher()
    core["plugin_mgr"].hook_on_startup(core)
    _print_startup_summary(core)


_bootstrap()


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def run(host: str = "0.0.0.0", port: int = 11435, debug: bool = False):
    """Dev entrypoint. En producción: gunicorn -w 1 -b 0.0.0.0:11435 api_server:app"""
    app_logger.info(f"[DEV] l3mcore API listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="l3mcore API Server (light)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=args.debug)
