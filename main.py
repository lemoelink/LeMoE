"""
l3mcore — Light Easy Mix Of Experts
Punto de entrada CLI: carga el router y los modelos especializados.
"""

import sys
import os
import re
import signal

from modules.logger import app_logger
from modules.config_manager import ConfigManager
from modules.router_factory import create_router
from modules.onnx_runner import SpecificModelRunner
from modules.ai_engine import AIEngine
from modules.expert_runner import ExpertDispatcher

_NON_PRINTABLE = re.compile(r'[\x00-\x1f\x7f]')


def _safe_log(text: str, max_len: int = 120) -> str:
    cleaned = _NON_PRINTABLE.sub(' ', text)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


class l3mcore:
    """
    Orquestador principal MoE.
    Flujo: texto -> Router -> Label -> ExpertDispatcher -> modelo local / API / Ollama
    Si el router devuelve 'null', se usa el motor GGUF como fallback.
    """

    def __init__(self):
        app_logger.info("Starting l3mcore (light)...")

        self.config     = ConfigManager()
        self.router     = create_router(self.config)
        self.runner     = SpecificModelRunner(
            models_base_path="models",
            stats_path="data/model_stats.json"
        )
        self.ai_engine  = AIEngine()
        self.dispatcher = ExpertDispatcher(self.runner, self.ai_engine)

        app_logger.info("l3mcore ready.")

    def process(self, text: str) -> str:
        """
        Procesa una entrada de texto:
        1. Sanitiza con el interceptor de seguridad.
        2. El router clasifica el texto y obtiene el label del modelo.
        3. Si hay label válido, ExpertDispatcher ejecuta la petición.
        4. Si no hay confianza suficiente, usa el motor GGUF como fallback.
        """
        if not text or not text.strip():
            return ""

        try:
            from modules.utils_text import sanitize
            intercepted = sanitize(text)
            if intercepted is not None:
                return intercepted
        except Exception as e:
            app_logger.warning(f"Security interceptor failed: {e}")

        label, score = self.router.predict(text)
        app_logger.info(f"Router: label='{label}' score={score:.3f}")

        result = None

        if label and label != "null":
            try:
                if hasattr(self.router, 'get_expert_config'):
                    cfg = self.router.get_expert_config(label)
                    if cfg:
                        result = self.dispatcher.run(text, cfg)
                if result is None:
                    cfg = {"type": "local", "format": "onnx", "label": label}
                    result = self.dispatcher.run(text, cfg)
            except Exception as e:
                app_logger.error(f"Error en experto ({label}): {e}. Usando GGUF fallback.")

        if result is None:
            result = self.ai_engine.generate_response(text)
            app_logger.info(f"AIEngine (GGUF) -> '{_safe_log(result)}'")

        return result

    def shutdown(self):
        app_logger.info("Shutting down l3mcore...")
        if hasattr(self.router, 'clear_cache'):
            self.router.clear_cache()
        app_logger.info("Shutdown complete.")


def _handle_signal(sig, frame, instance):
    instance.shutdown()
    sys.exit(0)


def main():
    core = l3mcore()

    signal.signal(signal.SIGINT,  lambda s, f: _handle_signal(s, f, core))
    signal.signal(signal.SIGTERM, lambda s, f: _handle_signal(s, f, core))

    app_logger.info("l3mcore waiting for input (stdin).")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        app_logger.info(f"stdin: '{_safe_log(line)}'")
        print(core.process(line), flush=True)

    core.shutdown()


if __name__ == "__main__":
    main()
