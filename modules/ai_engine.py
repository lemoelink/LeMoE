"""
AIEngine — Motor GGUF con llama-cpp-python

Carga lazy del modelo en primer uso.
TTL de 5 minutos: el modelo se descarga si lleva inactivo > TTL.

Fix de seguridad: capture local de self.llm bajo lock para evitar
TOCTOU entre _ensure_model_loaded() y generate_response().
"""
import gc
import os
import time
import threading
from modules.logger import app_logger

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    app_logger.warning("llama-cpp-python not installed. AIEngine disabled.")


class AIEngine:
    """
    Motor de inferencia para modelos GGUF (fallback general).

    Thread-safety:
      Todos los accesos a self.llm y self.is_ready están protegidos
      por self._llm_lock. El TTL cleanup y generate_response capturan
      self.llm en variable local bajo lock para evitar TOCTOU.
    """

    DEFAULT_GGUF = "models/gemma-2-2b-it-Q4_K_M.gguf"

    def __init__(self, model_path: str | None = None, config_manager=None):
        self._config_manager = config_manager
        self._llm_lock       = threading.Lock()   # Protege llm + is_ready
        self._stop_cleanup   = False

        # Resolución del path del modelo
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists("models/gemma-2-2b-it-Q8_0.gguf"):
            self.model_path = "models/gemma-2-2b-it-Q8_0.gguf"
        else:
            self.model_path = self.DEFAULT_GGUF

        self.llm       = None
        self.is_ready  = False
        self.last_access = 0.0
        self.ttl_seconds = 300  # 5 minutos

        app_logger.info(f"AIEngine configured: {self.model_path} (lazy load)")

        if LLAMA_AVAILABLE:
            t = threading.Thread(
                target=self._ttl_cleanup_loop,
                daemon=True,
                name="AIEngine_TTL_Cleanup",
            )
            t.start()

    def _llm_config(self) -> dict:
        if self._config_manager is None:
            return {}
        try:
            return self._config_manager.get("ai_engine", {}) or {}
        except Exception:
            return {}

    def _ttl_cleanup_loop(self):
        while not self._stop_cleanup:
            time.sleep(60)
            with self._llm_lock:
                if (
                    self.llm is not None
                    and self.last_access > 0
                    and (time.time() - self.last_access) > self.ttl_seconds
                ):
                    app_logger.info(
                        f"AIEngine TTL: unloading model (idle {self.ttl_seconds}s)"
                    )
                    self.llm      = None
                    self.is_ready = False
                    gc.collect()

    def _load_model_locked(self):
        """
        Carga el modelo GGUF. Debe llamarse con self._llm_lock adquirido.
        """
        if not os.path.exists(self.model_path):
            app_logger.error(f"AIEngine: model not found at '{self.model_path}'")
            return

        try:
            cfg   = self._llm_config()
            n_ctx = int(cfg.get("n_ctx", 4096 if "llama-3" in self.model_path.lower() else 2048))
            n_threads = int(cfg.get("n_threads", 4))
            n_batch   = int(cfg.get("n_batch", 512))

            app_logger.info(
                f"AIEngine: loading {os.path.basename(self.model_path)} "
                f"(n_ctx={n_ctx}, n_threads={n_threads})..."
            )
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                use_mmap=True,
                verbose=False,
            )
            self.is_ready = True
            gc.collect()
            app_logger.info("AIEngine: model loaded.")
        except Exception as e:
            app_logger.error(f"AIEngine: error loading model: {e}")
            self.llm      = None
            self.is_ready = False

    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Genera una respuesta. Thread-safe: captura llm bajo lock.
        Devuelve siempre un string (nunca lanza excepciones al caller).
        """
        if not prompt or not isinstance(prompt, str):
            return ""

        if not LLAMA_AVAILABLE:
            return "GGUF model not available (llama-cpp-python not installed)."

        # Cargar si no está listo (bajo lock)
        with self._llm_lock:
            self.last_access = time.time()
            if self.llm is None:
                self._load_model_locked()
            # Capturar referencia local para evitar TOCTOU con TTL cleanup
            llm_ref   = self.llm
            is_ready  = self.is_ready

        if not is_ready or llm_ref is None:
            return "AI model is not available at the moment."

        try:
            output = llm_ref(
                prompt,
                max_tokens=max_tokens,
                stop=["<end_of_turn>", "<eos>"],
                echo=False,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            app_logger.error(f"AIEngine: inference error: {e}")
            return "Error generating response."

    def generate_response_stream(self, prompt: str, max_tokens: int = 150):
        """Genera respuesta en modo streaming."""
        if not LLAMA_AVAILABLE:
            yield "GGUF model not available."
            return

        with self._llm_lock:
            self.last_access = time.time()
            if self.llm is None:
                self._load_model_locked()
            llm_ref  = self.llm
            is_ready = self.is_ready

        if not is_ready or llm_ref is None:
            yield "AI model is not available."
            return

        try:
            stream = llm_ref(
                prompt,
                max_tokens=max_tokens,
                stop=["<end_of_turn>", "<eos>"],
                echo=False,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=True,
            )
            for output in stream:
                yield output["choices"][0]["text"]
        except Exception as e:
            app_logger.error(f"AIEngine: stream error: {e}")
            yield " Error."

    def shutdown(self):
        """Libera el modelo de memoria."""
        self._stop_cleanup = True
        with self._llm_lock:
            if self.llm is not None:
                self.llm      = None
                self.is_ready = False
                gc.collect()
                app_logger.info("AIEngine: model unloaded.")


# Alias de compatibilidad
GemmaEngine = AIEngine
