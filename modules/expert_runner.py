import os
import gc
import json
import time
import ipaddress
import threading
import urllib.request
import urllib.error
from urllib.parse import urlparse
from modules.logger import app_logger

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    app_logger.warning("litellm is not installed. External API calls may fail.")


_ALLOWED_SCHEMES = {"http", "https"}

# Cloud metadata / link-local ranges always blocked
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/GCP/Azure metadata + link-local
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
]

# Default Ollama hostname allowlist. Add more entries in config.json under
# expert_runner.ollama_allowed_hosts if needed.
_DEFAULT_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}

_DEFAULT_API_TIMEOUT = 60  # seconds


def _get_runner_config(config_manager=None) -> dict:
    if config_manager is None:
        return {}
    return config_manager.get("expert_runner", {})


def _validate_ollama_url(url: str, allowed_hosts: set | None = None) -> str:
    """
    Validates an Ollama endpoint URL.

    - Only http/https schemes are accepted.
    - Cloud metadata IP ranges (169.254.x.x etc.) are always blocked.
    - Hostname validation: if the value resolves to an IP it is checked against
      the blocked networks; if it is a plain hostname it must be in allowed_hosts.
    - Private/loopback IPs are allowed by default.

    Raises ValueError on invalid URLs.
    """
    if allowed_hosts is None:
        allowed_hosts = _DEFAULT_ALLOWED_HOSTS

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Malformed Ollama URL: {url}") from exc

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsafe URL scheme '{parsed.scheme}' in Ollama config. "
            f"Only {_ALLOWED_SCHEMES} are allowed."
        )

    hostname = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(hostname)
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(f"Ollama URL points to a blocked network ({net}): {url}")
    except ValueError as exc:
        if "blocked network" in str(exc) or "scheme" in str(exc):
            raise
        # It's a plain hostname — check against the allowlist
        if hostname not in allowed_hosts:
            raise ValueError(
                f"Ollama hostname '{hostname}' is not in the allowed hosts list. "
                f"Add it to expert_runner.ollama_allowed_hosts in config.json."
            )

    return url


def _extract_text_from_messages(messages) -> str:
    """Extracts a plain text string from a messages list for local model inference."""
    if isinstance(messages, str):
        return messages

    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    if part.get("type", "text") == "text":
                        parts.append(str(part.get("text", part.get("content", ""))))
    return " ".join(parts)


_SYS_PROMPT_MAX = 4000  # characters — hard cap before injecting into messages


def _inject_system_prompt(messages, expert_config: dict) -> list:
    """
    Prepends a system message from the expert's 'system_prompt' field.

    Rules:
    - Only acts when 'system_prompt' is a non-empty string in expert_config.
    - Truncated to _SYS_PROMPT_MAX characters before use.
    - If a system message already exists at index 0 it is left intact and the
      expert prompt is prepended before it, so user-supplied system context
      is never silently discarded.
    - Returns a new list; the original messages list is never mutated.
    """
    raw = expert_config.get("system_prompt", "")
    if not isinstance(raw, str) or not raw.strip():
        return messages if isinstance(messages, list) else list(messages)

    prompt = raw.strip()[:_SYS_PROMPT_MAX]
    msgs = list(messages) if isinstance(messages, list) else []
    system_msg = {"role": "system", "content": prompt}

    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        return [system_msg] + msgs
    return [system_msg] + msgs


def _notify_telemetry(expert_label: str, latency_ms: float, prompt_tokens: int, completion_tokens: int, success: bool = True) -> None:
    """Pushes detailed telemetry data to the telemetry plugin if loaded."""
    try:
        import sys
        m = sys.modules.get("l3mcore_plugin.telemetry_dashboard")
        if m and hasattr(m, "record_telemetry"):
            m.record_telemetry(expert_label, latency_ms, prompt_tokens, completion_tokens, success)
    except Exception:
        pass


class ExpertDispatcher:
    """
    Routes inference to the correct backend based on the expert config dict.

    Supported backends:
      'api'    -> External REST API via litellm (OpenAI, Anthropic, Groq, ...).
                  API key is read from the environment variable named in api_key_env.
      'ollama' -> Local or remote Ollama instance.
                  URL is validated for scheme and blocked networks before each call.
      'local'  -> Local ONNX model (via SpecificModelRunner) or GGUF (via AIEngine).
    """

    def __init__(self, onnx_runner, ai_engine, config_manager=None):
        self.onnx_runner = onnx_runner
        self.ai_engine = ai_engine
        self._config_manager = config_manager
        self._gguf_lock = threading.Lock()

    def _runner_cfg(self) -> dict:
        return _get_runner_config(self._config_manager)

    def run(self, messages, expert_config: dict, tools: list | None = None) -> str | dict:
        """
        Runs inference for the given expert.
        tools: optional list of OpenAI-format tool definitions.
               Only passed to api/ollama experts; local models ignore it.
        Returns a string response, or a dict with 'tool_calls' if the model
        requests tool execution (only possible when tools is not None).
        """
        expert_type = expert_config.get('type', 'local').lower()
        messages = _inject_system_prompt(messages, expert_config)
        t0 = time.monotonic()
        prompt_tokens = 0
        completion_tokens = 0
        success = True
        try:
            if expert_type == 'api':
                result, prompt_tokens, completion_tokens = self._run_api(messages, expert_config, tools=tools)
            elif expert_type == 'ollama':
                result, prompt_tokens, completion_tokens = self._run_ollama(messages, expert_config, tools=tools)
            elif expert_type == 'local':
                # Modelos locales no soportan tool calling; tools se ignora
                result = self._run_local(messages, expert_config)
                prompt_tokens = int(len(_extract_text_from_messages(messages).split()) * 1.3)
                completion_tokens = int(len(result.split()) * 1.3)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")

            latency_ms = (time.monotonic() - t0) * 1000
            _notify_telemetry(expert_config.get('label', 'unknown'), latency_ms, prompt_tokens, completion_tokens, success=True)
            return result
        except Exception as e:
            success = False
            latency_ms = (time.monotonic() - t0) * 1000
            _notify_telemetry(expert_config.get('label', 'unknown'), latency_ms, prompt_tokens, completion_tokens, success=False)
            app_logger.error(f"Error executing expert '{expert_config.get('label')}': {e}")
            raise

    def _run_api(self, messages, config: dict, tools: list | None = None) -> tuple:
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm required for 'api' type experts")

        provider = config.get('provider', '')
        model_name = config.get('model_name', '')
        if not model_name:
            raise ValueError("model_name required for 'api' expert")

        litellm_model = f"{provider}/{model_name}" if provider and provider != 'openai' else model_name

        env_var = config.get('api_key_env', '')
        api_key = os.environ.get(env_var) if env_var else None
        if not api_key:
            app_logger.warning(f"API key not found in env var '{env_var}'. litellm will try its defaults.")

        cfg = self._runner_cfg()
        timeout = cfg.get("api_timeout", _DEFAULT_API_TIMEOUT)

        app_logger.info(f"ExpertDispatcher [api]: calling {litellm_model} (timeout={timeout}s)")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        formatted_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                formatted_messages.append(msg)
                continue

            new_msg = {"role": msg.get("role")}
            content = msg.get("content")
            images = msg.get("images")

            # Pasar tool_call_id y tool_calls tal cual para mensajes de herramientas
            if msg.get("role") == "tool":
                new_msg["content"] = str(content) if content is not None else ""
                if "tool_call_id" in msg:
                    new_msg["tool_call_id"] = msg["tool_call_id"]
                formatted_messages.append(new_msg)
                continue
            if msg.get("tool_calls"):
                new_msg["tool_calls"] = msg["tool_calls"]
                new_msg["content"] = content or ""
                formatted_messages.append(new_msg)
                continue

            if isinstance(content, list):
                new_msg["content"] = content
                formatted_messages.append(new_msg)
                continue

            if isinstance(images, list) and images:
                parts = []
                if content:
                    parts.append({"type": "text", "text": str(content)})
                for img in images:
                    if isinstance(img, str):
                        if not img.startswith("data:image/"):
                            img = f"data:image/png;base64,{img}"
                        parts.append({"type": "image_url", "image_url": {"url": img}})
                new_msg["content"] = parts
            else:
                new_msg["content"] = str(content) if content is not None else ""

            formatted_messages.append(new_msg)

        kwargs = {
            "model": litellm_model,
            "messages": formatted_messages,
            "api_key": api_key,
            "timeout": timeout,
        }
        if tools:
            kwargs["tools"] = tools

        response = litellm.completion(**kwargs)
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        choice = response.choices[0]
        # Si el modelo pide herramientas, devolver el dict de tool_calls
        if getattr(choice, 'finish_reason', '') == 'tool_calls':
            return {"tool_calls": choice.message.tool_calls}, prompt_tokens, completion_tokens

        return choice.message.content.strip(), prompt_tokens, completion_tokens

    def _run_ollama(self, messages, config: dict, tools: list | None = None) -> tuple:
        raw_url = config.get('url', 'http://127.0.0.1:11434').rstrip('/')
        model_name = config.get('model_name', 'llama3')

        cfg = self._runner_cfg()
        allowed_hosts = set(cfg.get("ollama_allowed_hosts", [])) | _DEFAULT_ALLOWED_HOSTS
        timeout = cfg.get("ollama_timeout", _DEFAULT_API_TIMEOUT)

        url = _validate_ollama_url(raw_url, allowed_hosts=allowed_hosts)
        endpoint = f"{url}/api/chat"
        app_logger.info(f"ExpertDispatcher [ollama]: POST {endpoint} ({model_name})")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        formatted_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                formatted_messages.append(msg)
                continue

            new_msg = {
                "role": msg.get("role"),
            }

            content = msg.get("content")
            images = msg.get("images") or []
            if not isinstance(images, list):
                images = [images]
            else:
                images = list(images)

            clean_images = []
            for img in images:
                if isinstance(img, str):
                    if img.startswith("data:image/") and ";base64," in img:
                        img = img.split(";base64,", 1)[1]
                    clean_images.append(img)

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type == "text":
                            text_parts.append(part.get("text", ""))
                        elif part_type == "image_url":
                            img_url_dict = part.get("image_url")
                            if isinstance(img_url_dict, dict):
                                img_url = img_url_dict.get("url", "")
                                if isinstance(img_url, str) and img_url.startswith("data:image/"):
                                    if ";base64," in img_url:
                                        raw_b64 = img_url.split(";base64,", 1)[1]
                                        clean_images.append(raw_b64)
                new_msg["content"] = "\n".join(text_parts)
            else:
                new_msg["content"] = str(content) if content is not None else ""

            if clean_images:
                new_msg["images"] = clean_images

            formatted_messages.append(new_msg)

        data = {"model": model_name, "messages": formatted_messages, "stream": False}
        if tools:
            data["tools"] = tools

        app_logger.info(f"ExpertDispatcher [ollama]: {len(formatted_messages)} msgs, roles={[m.get('role') for m in formatted_messages]}, tools={len(tools) if tools else 0}")
        if formatted_messages:
            first = formatted_messages[0]
            app_logger.info(f"ExpertDispatcher [ollama]: msg[0] role={first.get('role')} content={str(first.get('content',''))[:150]}")

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                prompt_tokens = result.get('prompt_eval_count', 0)
                completion_tokens = result.get('eval_count', 0)
                message = result.get('message', {})

                # Ollama devuelve tool_calls dentro de message si el modelo los solicita
                if message.get('tool_calls'):
                    return {"tool_calls": message['tool_calls']}, prompt_tokens, completion_tokens

                return message.get('content', '').strip(), prompt_tokens, completion_tokens
        except urllib.error.URLError as e:
            raise RuntimeError(f"Error connecting to Ollama at {url}: {e}")

    def _run_local(self, messages, config: dict) -> str:
        model_format = config.get('format', 'onnx').lower()
        text = _extract_text_from_messages(messages)
        label = config.get('label', '')
        model_path = config.get('model_path')

        if model_format == 'onnx':
            return self.onnx_runner.generate_command(text, label, model_path)

        elif model_format == 'gguf':
            with self._gguf_lock:
                original_path = self.ai_engine.model_path
                try:
                    if model_path and os.path.exists(model_path):
                        self.ai_engine.model_path = model_path
                        if getattr(self.ai_engine, 'llm', None):
                            self.ai_engine.llm = None
                            gc.collect()
                    return self.ai_engine.generate_response(text)
                finally:
                    self.ai_engine.model_path = original_path

        elif model_format == 'huggingface':
            raise NotImplementedError("Local 'huggingface' format not implemented yet.")

        else:
            raise ValueError(f"Unknown local format: {model_format}")
