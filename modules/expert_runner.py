"""
l3mcore ExpertDispatcher — versión light

Backends soportados:
  'ollama' -> Ollama local/remoto (urllib nativo)
  'api'    -> API OpenAI-compatible via urllib (OpenAI, Groq, Mistral, Together...)
  'local'  -> ONNX (SpecificModelRunner) o GGUF (AIEngine)

litellm eliminado. Solo modelos locales u Ollama en la versión light.
"""
import gc
import json
import os
import time
import ipaddress
import threading
import urllib.request
import urllib.error
from urllib.parse import urlparse
from modules.logger import app_logger


_ALLOWED_SCHEMES = {"http", "https"}

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/GCP/Azure metadata + link-local
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
]

_DEFAULT_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}
_DEFAULT_API_TIMEOUT = 60  # seconds


def _get_runner_config(config_manager=None) -> dict:
    if config_manager is None:
        return {}
    return config_manager.get("expert_runner", {})


def _validate_ollama_url(url: str, allowed_hosts: set | None = None) -> str:
    """Valida URL Ollama: solo http/https, sin redes cloud-metadata."""
    if allowed_hosts is None:
        allowed_hosts = _DEFAULT_ALLOWED_HOSTS

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Malformed URL: {url}") from exc

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Unsafe URL scheme '{parsed.scheme}'. Only {_ALLOWED_SCHEMES} allowed.")

    hostname = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(hostname)
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(f"URL points to a blocked network ({net}): {url}")
    except ValueError as exc:
        if "blocked network" in str(exc) or "scheme" in str(exc):
            raise
        if hostname not in allowed_hosts:
            raise ValueError(
                f"Hostname '{hostname}' not in allowed hosts. "
                "Add it to expert_runner.ollama_allowed_hosts in config.json."
            )

    return url


def _extract_text_from_messages(messages) -> str:
    """Extrae texto plano de una lista de mensajes."""
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


_SYS_PROMPT_MAX = 4000


def _inject_system_prompt(messages, expert_config: dict) -> list:
    """Prepende un system message del campo 'system_prompt' del experto."""
    raw = expert_config.get("system_prompt", "")
    if not isinstance(raw, str) or not raw.strip():
        return messages if isinstance(messages, list) else list(messages)

    prompt = raw.strip()[:_SYS_PROMPT_MAX]
    msgs = list(messages) if isinstance(messages, list) else []
    system_msg = {"role": "system", "content": prompt}

    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        return [system_msg] + msgs
    return [system_msg] + msgs


def _format_messages_for_api(messages) -> list:
    """Convierte mensajes al formato OpenAI-compatible."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    formatted = []
    for msg in messages:
        if not isinstance(msg, dict):
            formatted.append(msg)
            continue

        new_msg = {"role": msg.get("role")}
        content = msg.get("content")
        images = msg.get("images")

        if isinstance(content, list):
            new_msg["content"] = content
        elif isinstance(images, list) and images:
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

        formatted.append(new_msg)
    return formatted


class ExpertDispatcher:
    """
    Enruta la inferencia al backend correcto según el tipo de experto:
      'api'    -> API OpenAI-compatible via urllib (sin litellm)
      'ollama' -> Ollama local/remoto via urllib
      'local'  -> ONNX (SpecificModelRunner) o GGUF (AIEngine)
    """

    def __init__(self, onnx_runner, ai_engine, config_manager=None):
        self.onnx_runner = onnx_runner
        self.ai_engine = ai_engine
        self._config_manager = config_manager
        self._gguf_lock = threading.Lock()

    def _runner_cfg(self) -> dict:
        return _get_runner_config(self._config_manager)

    def run(self, messages, expert_config: dict) -> str:
        """
        Ejecuta la inferencia para el experto dado.
        Devuelve siempre un string con la respuesta.
        """
        expert_type = expert_config.get('type', 'local').lower()
        messages = _inject_system_prompt(messages, expert_config)

        try:
            if expert_type == 'api':
                return self._run_api(messages, expert_config)
            elif expert_type == 'ollama':
                return self._run_ollama(messages, expert_config)
            elif expert_type == 'local':
                return self._run_local(messages, expert_config)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
        except Exception as e:
            app_logger.error(f"Error executing expert '{expert_config.get('label')}': {e}")
            raise

    def _run_api(self, messages, config: dict) -> str:
        """Llamada a API OpenAI-compatible via urllib (sin litellm)."""
        base_url = config.get('url', 'https://api.openai.com').rstrip('/')
        model_name = config.get('model_name', '')
        if not model_name:
            raise ValueError("model_name required for 'api' expert")

        env_var = config.get('api_key_env', '')
        api_key = os.environ.get(env_var) if env_var else None

        cfg = self._runner_cfg()
        timeout = cfg.get("api_timeout", _DEFAULT_API_TIMEOUT)

        endpoint = f"{base_url}/v1/chat/completions"
        app_logger.info(f"ExpertDispatcher [api]: POST {endpoint} ({model_name})")

        formatted = _format_messages_for_api(messages)
        payload = json.dumps({
            "model": model_name,
            "messages": formatted,
            "stream": False,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(endpoint, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"API error {e.code} from {endpoint}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach API at {endpoint}: {e}")

    def _run_ollama(self, messages, config: dict) -> str:
        """Llamada a Ollama via urllib."""
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

        formatted = []
        for msg in messages:
            if not isinstance(msg, dict):
                formatted.append(msg)
                continue
            new_msg = {"role": msg.get("role")}
            content = msg.get("content")
            images = msg.get("images") or []
            clean_images = []
            for img in (images if isinstance(images, list) else [images]):
                if isinstance(img, str):
                    if img.startswith("data:image/") and ";base64," in img:
                        img = img.split(";base64,", 1)[1]
                    clean_images.append(img)

            if isinstance(content, list):
                new_msg["content"] = "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                new_msg["content"] = str(content) if content is not None else ""

            if clean_images:
                new_msg["images"] = clean_images
            formatted.append(new_msg)

        data = json.dumps({"model": model_name, "messages": formatted, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Error connecting to Ollama at {url}: {e}")

    def _run_local(self, messages, config: dict) -> str:
        """Ejecuta modelo local: ONNX o GGUF."""
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

        else:
            raise ValueError(f"Unknown local format: {model_format}")
