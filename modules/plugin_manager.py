"""
l3mcore PluginManager — versión light

Carga dinámica de plugins desde plugins/.py (carpeta vacía en light).
Los hooks son funcionales; si no hay plugins, todo es un no-op.
"""
import os
import re
import sys
import threading
import importlib.util
from .logger import app_logger

_PLUGIN_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


class PluginManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super(PluginManager, cls).__new__(cls)
                    instance._plugins = []
                    instance._plugin_accepts_label = []
                    instance._load_plugins()
                    cls._instance = instance
        return cls._instance

    def _load_plugins(self):
        plugin_dir = os.path.join(os.getcwd(), 'plugins')
        if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
            app_logger.debug("Plugin directory not found. Plugins disabled.")
            return

        for filename in sorted(os.listdir(plugin_dir)):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue

            plugin_name = filename[:-3]
            if not _PLUGIN_NAME_RE.match(plugin_name):
                app_logger.warning(f"Plugin '{filename}' has invalid characters. Skipped.")
                continue

            namespace_key = f"l3mcore_plugin.{plugin_name}"
            if namespace_key in sys.modules:
                continue

            file_path = os.path.join(plugin_dir, filename)
            try:
                spec = importlib.util.spec_from_file_location(namespace_key, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[namespace_key] = module
                    spec.loader.exec_module(module)
                    self._plugins.append(module)

                    accepts_label = False
                    if hasattr(module, 'after_generation'):
                        try:
                            import inspect
                            sig = inspect.signature(module.after_generation)
                            accepts_label = len(sig.parameters) >= 2
                        except Exception:
                            accepts_label = False
                    self._plugin_accepts_label.append(accepts_label)
                    app_logger.info(f"Loaded plugin: {plugin_name}")
            except Exception as e:
                sys.modules.pop(namespace_key, None)
                app_logger.error(f"Failed to load plugin {plugin_name}: {e}")

    # --- Hooks -----------------------------------------------------------

    def hook_before_routing(self, prompt: str) -> str:
        for plugin in self._plugins:
            if hasattr(plugin, 'before_routing'):
                try:
                    result = plugin.before_routing(prompt)
                    if isinstance(result, str):
                        prompt = result
                except Exception as e:
                    app_logger.error(f"Plugin {plugin.__name__} (before_routing): {e}")
        return prompt

    def hook_override_route(self, messages: list) -> str | None:
        for plugin in self._plugins:
            if hasattr(plugin, 'override_route'):
                try:
                    target = plugin.override_route(messages)
                    if target:
                        return target
                except Exception as e:
                    app_logger.error(f"Plugin {plugin.__name__} (override_route): {e}")
        return None

    def hook_before_expert(self, messages: list, expert_config: dict) -> None:
        for plugin in self._plugins:
            if hasattr(plugin, 'before_expert'):
                try:
                    plugin.before_expert(messages, expert_config)
                except Exception as e:
                    app_logger.error(f"Plugin {plugin.__name__} (before_expert): {e}")

    def hook_after_generation(self, response: str, expert_label: str = None) -> str:
        for i, plugin in enumerate(self._plugins):
            if hasattr(plugin, 'after_generation'):
                try:
                    if self._plugin_accepts_label[i]:
                        result = plugin.after_generation(response, expert_label)
                    else:
                        result = plugin.after_generation(response)
                    if isinstance(result, str):
                        response = result
                except Exception as e:
                    app_logger.error(f"Plugin {plugin.__name__} (after_generation): {e}")
        return response

    def hook_before_request(self, request):
        for plugin in self._plugins:
            if hasattr(plugin, 'before_request'):
                try:
                    result = plugin.before_request(request)
                    if result is not None:
                        return result
                except Exception as e:
                    app_logger.error(f"Plugin {plugin.__name__} (before_request): {e}")
        return None

    def hook_on_startup(self, core_context: dict):
        for plugin in self._plugins:
            if hasattr(plugin, 'on_startup'):
                try:
                    plugin.on_startup(core_context)
                except Exception as e:
                    app_logger.error(f"Plugin {plugin.__name__} (on_startup): {e}")

    def hook_on_expert_failure(self, label: str, reason: str):
        for plugin in self._plugins:
            if hasattr(plugin, 'on_expert_failure'):
                try:
                    plugin.on_expert_failure(label, reason)
                except Exception as e:
                    app_logger.error(f"Plugin {plugin.__name__} (on_expert_failure): {e}")
