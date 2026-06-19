"""
l3mcore ExpertDispatcher — versión light

Backends:
  'ollama' -> Ollama local/remoto via urllib nativo
  'api'    -> API OpenAI-compatible via urllib nativo
  'local'  -> ONNX (SpecificModelRunner) o GGUF (AIEngine)

Diseñado para entornos embedidos/robóticos: sin dependencias pesadas,
timeouts explícitos, validación de URLs, thread-safe.
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
    ipaddress.ip_network("169.254.0.0/16"),  # Cloud metadata + link-local
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
]

_DEFAULT_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DEFAULT_API_TIMEOUT   = 60   # segundos
_SYS_PROMPT_MAX        = 4000 # caracteres máx. para system prompts


def _get_runner_config(config_manager) -> dict:
    if config_manager is None:
        return {}
    try:
        return config_manager.get("expert_runner", {}) or {}
    except Exception:
        return {}


def _validate_ollama_url(url: str, allowed_hosts: frozenset | set | None = None) -> str:
    """
    Valida una URL Ollama:
    - Solo http/https.
    - Bloquea rangos cloud-metadata (169.254.x.x, 100.64.x.x).
    - Hostname debe estar en allowed_hosts (por defecto: localhost/127.0.0.1/::1).
    Lanza ValueError en caso de URL inválida o no permitida.
    """
    if allowed_hosts is None:
        allowed_hosts = _DEFAULT_ALLOWED_HOSTS

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Malformed URL '{url}'") from exc

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsafe URL scheme '{parsed.scheme}'. Allowed: {_ALLOWED_SCHEMES}"
        )

    hostname = (parsed.hostname or "").lower()
    try:
        addr = ipaddress.ip_address(hostname)
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(f"URL targets a blocked network ({net}): {url}")
    except ValueError as exc:
        if "blocked network" in str(exc):
            raise
        # Nombre de host (no IP) — verificar lista de permitidos
        if hostname not in allowed_hosts:
            raise ValueError(
                f"Hostname '{hostname}' not in allowed_hosts. "
                "Add it to expert_runner.ollama_allowed_hosts in config.json."
            )

    return url


def _extract_text(messages) -> str:
    """
    Extrae texto plano de un string o lista de mensajes.
    Siempre devuelve un string (nunca None).
    """
    if isinstance(messages, str):
        return messages.strip()

    if not isinstance(messages, list):
        return ""

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
                elif isinstance(part, dict) and part.get("type", "text") == "text":
                    parts.append(str(part.get("text", part.get("content", ""))))
    return " ".join(parts).strip()


def _inject_system_prompt(messages, expert_config: dict) -> list:
    """
    Prepende un system message con el system_prompt del experto.
    Siempre devuelve una lista de mensajes bien formada.
    Nunca muta el input original.
    """
    raw = expert_config.get("system_prompt", "")
    if not isinstance(raw, str) or not raw.strip():
        # Sin system prompt: devolver copia de la lista original
        if isinstance(messages, list):
            return list(messages)
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return []

    prompt = raw.strip()[:_SYS_PROMPT_MAX]
    system_msg = {"role": "system", "content": prompt}

    if isinstance(messages, str):
        # Convertir string a lista bien formada
        return [system_msg, {"role": "user", "content": messages}]

    msgs = list(messages) if isinstance(messages, list) else []
    # Si ya hay un system message, preponer el nuestro antes (mayor prioridad)
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        return [system_msg] + msgs
    return [system_msg] + msgs


def _format_messages_openai(messages) -> list:
    """
    Normaliza mensajes al formato OpenAI-compatible.
    Maneja strings, listas y contenidos mixtos (texto + imágenes base64).
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if not isinstance(messages, list):
        return []

    formatted = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role    = msg.get("role", "user")
        content = msg.get("content")
        images  = msg.get("images")

        new_msg = {"role": role}

        if isinstance(content, list):
            # Ya es contenido multimodal
            new_msg["content"] = content
        elif isinstance(images, list) and images:
            # Texto + imágenes base64
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


def _http_post_json(url: str, payload: dict, headers: dict,
                    timeout: int) -> dict:
    """
    Realiza un POST JSON via urllib y devuelve el body parseado.
    Lanza RuntimeError con mensaje limpio (sin paths internos) en caso de fallo.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} from API: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach endpoint: {e.reason}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from API: {e}")


class ExpertDispatcher:
    """
    Enruta la inferencia al backend correcto según el tipo de experto.

    Backends soportados:
      'api'    -> API OpenAI-compatible (urllib nativo, sin litellm)
      'ollama' -> Ollama local o remoto (urllib nativo)
      'local'  -> ONNX (SpecificModelRunner) o GGUF (AIEngine)

    Thread-safe: el lock de GGUF evita llamadas concurrentes al mismo modelo.
    Stateless por petición: no acumula estado entre llamadas.
    """

    def __init__(self, onnx_runner, ai_engine, config_manager=None):
        self.onnx_runner      = onnx_runner
        self.ai_engine        = ai_engine
        self._config_manager  = config_manager
        self._gguf_lock       = threading.Lock()

    def _runner_cfg(self) -> dict:
        return _get_runner_config(self._config_manager)

    def run(self, messages, expert_config: dict) -> str:
        """
        Ejecuta la inferencia para el experto dado.
        - messages: str o list[dict] con formato OpenAI/Ollama.
        - expert_config: dict con 'type', 'model_name', etc.
        - Devuelve siempre un string; lanza RuntimeError en caso de fallo.
        """
        if not isinstance(expert_config, dict):
            raise ValueError("expert_config must be a dict")

        expert_type = expert_config.get("type", "local").lower()
        # _inject_system_prompt garantiza que messages sea siempre list[dict]
        messages = _inject_system_prompt(messages, expert_config)

        try:
            if expert_type == "api":
                return self._run_api(messages, expert_config)
            elif expert_type == "ollama":
                return self._run_ollama(messages, expert_config)
            elif expert_type == "local":
                return self._run_local(messages, expert_config)
            else:
                raise ValueError(f"Unknown expert type: '{expert_type}'")
        except Exception as e:
            label = expert_config.get("label", "unknown")
            app_logger.error(f"ExpertDispatcher: expert '{label}' failed: {e}")
            raise

    # --- Backend: API OpenAI-compatible ------------------------------------

    def _run_api(self, messages: list, config: dict) -> str:
        """
        Llama a cualquier API con endpoint /v1/chat/completions (formato OpenAI).
        No depende de litellm. Soporta: OpenAI, Groq, Mistral, Together, etc.
        """
        base_url   = config.get("url", "https://api.openai.com").rstrip("/")
        model_name = config.get("model_name", "")
        if not model_name:
            raise ValueError("'model_name' is required for 'api' expert")

        env_var = config.get("api_key_env", "")
        api_key = os.environ.get(env_var) if env_var else None
        if not api_key:
            app_logger.warning(
                f"ExpertDispatcher [api]: env var '{env_var}' not set. "
                "Proceeding without auth (may fail)."
            )

        cfg     = self._runner_cfg()
        timeout = int(cfg.get("api_timeout", _DEFAULT_API_TIMEOUT))

        endpoint  = f"{base_url}/v1/chat/completions"
        formatted = _format_messages_openai(messages)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model":    model_name,
            "messages": formatted,
            "stream":   False,
        }

        app_logger.info(
            f"ExpertDispatcher [api]: {model_name} @ {base_url} "
            f"({len(formatted)} msgs, timeout={timeout}s)"
        )

        result = _http_post_json(endpoint, payload, headers, timeout)

        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected API response structure: {e}")

    # --- Backend: Ollama --------------------------------------------------

    def _run_ollama(self, messages: list, config: dict) -> str:
        """Llama a una instancia Ollama via su API REST nativa."""
        raw_url    = config.get("url", "http://127.0.0.1:11434").rstrip("/")
        model_name = config.get("model_name", "llama3")

        cfg           = self._runner_cfg()
        allowed_extra = set(cfg.get("ollama_allowed_hosts", []))
        allowed_hosts = _DEFAULT_ALLOWED_HOSTS | allowed_extra
        timeout       = int(cfg.get("ollama_timeout", _DEFAULT_API_TIMEOUT))

        url      = _validate_ollama_url(raw_url, allowed_hosts=allowed_hosts)
        endpoint = f"{url}/api/chat"

        # Normalizar mensajes al formato que Ollama espera
        formatted = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role    = msg.get("role", "user")
            content = msg.get("content")
            images  = msg.get("images") or []

            # Extraer imágenes base64 y quitar prefijo data:...
            clean_images = []
            for img in (images if isinstance(images, list) else [images]):
                if isinstance(img, str):
                    if img.startswith("data:image/") and ";base64," in img:
                        img = img.split(";base64,", 1)[1]
                    clean_images.append(img)

            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                text = "\n".join(text_parts)
            else:
                text = str(content) if content is not None else ""

            new_msg = {"role": role, "content": text}
            if clean_images:
                new_msg["images"] = clean_images
            formatted.append(new_msg)

        payload = {"model": model_name, "messages": formatted, "stream": False}
        headers = {"Content-Type": "application/json"}

        app_logger.info(
            f"ExpertDispatcher [ollama]: {model_name} @ {url} "
            f"({len(formatted)} msgs, timeout={timeout}s)"
        )

        result = _http_post_json(endpoint, payload, headers, timeout)

        try:
            return result.get("message", {}).get("content", "").strip()
        except (AttributeError, TypeError) as e:
            raise RuntimeError(f"Unexpected Ollama response structure: {e}")

    # --- Backend: Local (ONNX / GGUF) -------------------------------------

    def _run_local(self, messages: list, config: dict) -> str:
        """
        Ejecuta un modelo local:
        - 'onnx': SpecificModelRunner (T5 cuantizado, etc.)
        - 'gguf': AIEngine con llama-cpp-python (lazy load, thread-safe)
        """
        model_format = config.get("format", "onnx").lower()
        text         = _extract_text(messages)
        label        = config.get("label", "")
        model_path   = config.get("model_path")

        if not text:
            raise ValueError("Empty text for local model inference")

        if model_format == "onnx":
            return self.onnx_runner.generate_command(text, label, model_path)

        elif model_format == "gguf":
            # Lock garantiza que solo una petición usa el motor GGUF a la vez
            with self._gguf_lock:
                original_path = self.ai_engine.model_path
                try:
                    if model_path and os.path.exists(model_path):
                        if model_path != original_path:
                            self.ai_engine.model_path = model_path
                            # Forzar recarga si el path cambió
                            if getattr(self.ai_engine, "llm", None):
                                self.ai_engine.llm = None
                                gc.collect()
                    return self.ai_engine.generate_response(text)
                finally:
                    # Siempre restaurar el path original
                    if self.ai_engine.model_path != original_path:
                        self.ai_engine.model_path = original_path

        else:
            raise ValueError(
                f"Unknown local format '{model_format}'. Use 'onnx' or 'gguf'."
            )
