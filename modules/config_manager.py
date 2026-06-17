import json
import os
import threading
import logging
import time
import base64

logger = logging.getLogger("ConfigManager")

_CONFIG_FILE = 'config/config.json'


_REQUIRED_ROUTER_FIELDS = ['mode', 'router_type', 'categories_file']

def validate_config(config: dict) -> list[str]:
    """Validates required config fields. Returns list of warning strings."""
    warnings = []
    router = config.get('router', {})
    for field in _REQUIRED_ROUTER_FIELDS:
        if field not in router:
            warnings.append(f"router.{field} is missing (using default)")

    if router.get('confidence_threshold', 0.4) < 0 or router.get('confidence_threshold', 0.4) > 1:
        warnings.append("router.confidence_threshold should be between 0 and 1")

    rl = config.get('rate_limiting', {})
    if rl.get('max_requests', 60) < 1:
        warnings.append("rate_limiting.max_requests should be >= 1")

    return warnings


def deobfuscate_value(val):
    if isinstance(val, str):
        if val.startswith("env:"):
            env_var = val[4:]
            return os.getenv(env_var, "")
        elif val.startswith("base64:"):
            try:
                decoded = base64.b64decode(val[7:]).decode("utf-8")
                return decoded
            except Exception as e:
                logger.error(f"Error decoding base64 value '{val}': {e}")
                return val
        elif val.startswith("obfuscated:"):
            try:
                decoded = base64.b64decode(val[11:]).decode("utf-8")
                return decoded
            except Exception as e:
                logger.error(f"Error decoding obfuscated value '{val}': {e}")
                return val
    elif isinstance(val, dict):
        return {k: deobfuscate_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [deobfuscate_value(v) for v in val]
    return val


class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super(ConfigManager, cls).__new__(cls)
                    instance._config = {}
                    instance._deob_cache = {}
                    instance._file_mtime = 0.0
                    instance._data_lock = threading.Lock()
                    instance.load()
                    cls._instance = instance
        return cls._instance

    def load(self):
        with self._data_lock:
            try:
                if os.path.exists(_CONFIG_FILE):
                    with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                        self._config = loaded if loaded is not None else {}
                    self._file_mtime = os.path.getmtime(_CONFIG_FILE)
                    self._deob_cache.clear()
                    warnings = validate_config(self._config)
                    for w in warnings:
                        logger.warning(f"Config: {w}")
                else:
                    logger.warning(f"Configuration file {_CONFIG_FILE} not found. Using defaults.")
                    self._config = {}
                    self._file_mtime = 0.0
                    self._deob_cache.clear()
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self._config = {}
                self._deob_cache.clear()

    def save(self):
        with self._data_lock:
            try:
                tmp_path = _CONFIG_FILE + ".tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=4)
                os.replace(tmp_path, _CONFIG_FILE)
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")

    def get(self, key, default=None):
        with self._data_lock:
            if key in self._deob_cache:
                return self._deob_cache[key]
            val = self._config.get(key, default)
            result = deobfuscate_value(val)
            self._deob_cache[key] = result
            return result

    def set(self, key, value):
        with self._data_lock:
            self._config[key] = value
            self._deob_cache.clear()
        self.save()

    def get_all(self):
        with self._data_lock:
            return deobfuscate_value(self._config)

    def check_for_changes(self):
        """Logs a warning if config.json on disk is newer than the loaded version."""
        try:
            if os.path.exists(_CONFIG_FILE):
                current_mtime = os.path.getmtime(_CONFIG_FILE)
                if current_mtime > self._file_mtime:
                    age = int(current_mtime - self._file_mtime)
                    logger.warning(
                        f"config.json has been modified on disk ({age}s ago) but the running "
                        "instance still uses the old version. Restart the server to apply changes."
                    )
        except Exception as e:
            logger.debug(f"ConfigManager.check_for_changes: {e}")
