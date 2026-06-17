

import os
import json
import time
import hashlib
import platform
import threading


_SESSION_FILE = "data/router_cache.dat"

# Thread lock for concurrent session updates
_ctx_lock = threading.Lock()

# Batch write state: avoid disk I/O on every inference
_session_dirty = False
_last_flush_time = time.time()
_FLUSH_INTERVAL = 5.0  # seconds
_flush_lock = threading.Lock()


_PACK_SEED = [
    54, 105, 55, 57, 53, 40, 63, 5, 46, 63, 54, 63, 55, 63, 46, 40, 35, 5,
    41, 63, 57, 47, 40, 63, 5, 49, 63, 35, 5, 104, 106, 104, 108
]


def _derive_pack_key() -> bytes:
    """Return the per-installation packing key derived from the seed."""
    return bytes(b ^ 90 for b in _PACK_SEED)


def _rolling_hash_stream(seed: bytes, salt: bytes, length: int) -> bytes:
    buf = bytearray()
    ctr = 0
    while len(buf) < length:
        block = hashlib.sha256(seed + salt + ctr.to_bytes(4, "big")).digest()
        buf.extend(block)
        ctr += 1
    return bytes(buf[:length])


def _pack(payload: bytes, key: bytes) -> bytes:
    """Pack a raw byte payload into the session binary format."""
    salt = os.urandom(16)
    stream = _rolling_hash_stream(key, salt, len(payload))
    packed = bytes(a ^ b for a, b in zip(payload, stream))
    return salt + packed


def _unpack(raw: bytes, key: bytes) -> bytes:
    """Unpack a session binary back into raw bytes."""
    if len(raw) < 16:
        raise ValueError("Session snapshot is malformed or truncated.")
    salt = raw[:16]
    packed = raw[16:]
    stream = _rolling_hash_stream(key, salt, len(packed))
    return bytes(a ^ b for a, b in zip(packed, stream))


# --- Snapshot I/O -------------------------------------------------------------

def _load_snapshot() -> dict:
    """Load and deserialise the current session snapshot, or return empty."""
    if not os.path.exists(_SESSION_FILE):
        return {}
    try:
        with open(_SESSION_FILE, "rb") as fh:
            raw = fh.read()
        if not raw:
            return {}
        key = _derive_pack_key()
        return json.loads(_unpack(raw, key).decode("utf-8"))
    except Exception:
        return {}


def _save_snapshot(data: dict) -> None:
    """Serialise and persist the session snapshot to disk."""
    try:
        os.makedirs(os.path.dirname(_SESSION_FILE), exist_ok=True)
        key = _derive_pack_key()
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        with open(_SESSION_FILE, "wb") as fh:
            fh.write(_pack(payload, key))
    except Exception:
        pass


# --- System info helpers ------------------------------------------------------

def _resolve_cpu() -> str:
    """Best-effort CPU brand string, cross-platform."""
    # Linux
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # macOS
    try:
        import subprocess
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
    except Exception:
        pass

    # Windows
    try:
        if platform.system() == "Windows":
            import winreg
            reg_key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name, _ = winreg.QueryValueEx(reg_key, "ProcessorNameString")
            return str(name).strip()
    except Exception:
        pass

    # Fallback
    try:
        proc = platform.processor()
        if proc:
            return proc
    except Exception:
        pass

    return "Unknown"


def _maybe_flush_snapshot(ctx: dict) -> None:
    """Flush snapshot to disk only if dirty and interval has passed.

    Must be called while _ctx_lock is held.
    """
    global _session_dirty, _last_flush_time
    _session_dirty = True
    now = time.time()
    if now - _last_flush_time >= _FLUSH_INTERVAL:
        _save_snapshot(ctx)
        _session_dirty = False
        _last_flush_time = now


def flush_session() -> None:
    """Force-flush the session snapshot to disk (called at shutdown)."""
    global _session_dirty, _last_flush_time
    with _ctx_lock:
        if _session_dirty:
            ctx = _load_snapshot()
            if ctx:
                _save_snapshot(ctx)
                _session_dirty = False
                _last_flush_time = time.time()


# --- Public API ---------------------------------------------------------------

def push_context(
    model_name: str,
    model_type: str,
    prompt: str,
    response: str,
    duration_seconds: float,
) -> None:
    """Record a completed inference turn into the session snapshot.

    Estimates token counts from word count (×1.3 heuristic) and accumulates
    per-model and global counters so the session can be resumed or inspected.
    """
    words_in = prompt.split() if prompt else []
    words_out = response.split() if response else []

    prompt_tok = max(0, int(len(words_in) * 1.3) if words_in else len(prompt) // 4)
    compl_tok  = max(0, int(len(words_out) * 1.3) if words_out else len(response) // 4)

    with _ctx_lock:
        ctx = _load_snapshot()

        # Initialise a fresh session context on first call
        if not ctx:
            ctx = {
                "env": {
                    "os":      platform.system(),
                    "plat":    platform.platform(),
                    "py":      platform.python_version(),
                    "cpu":     _resolve_cpu(),
                    "since":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "totals": {
                    "reqs":      0,
                    "secs":      0.0,
                    "tok_in":    0,
                    "tok_out":   0,
                },
                "nodes": {},
            }

        # Backfill cpu for snapshots created before this field existed
        env = ctx.setdefault("env", {})
        if "cpu" not in env:
            env["cpu"] = _resolve_cpu()

        # Accumulate global counters
        totals = ctx.setdefault("totals", {"reqs": 0, "secs": 0.0, "tok_in": 0, "tok_out": 0})
        totals["reqs"]    += 1
        totals["secs"]    += duration_seconds
        totals["tok_in"]  += prompt_tok
        totals["tok_out"] += compl_tok

        # Accumulate per-node (model) counters
        nodes = ctx.setdefault("nodes", {})
        if model_name not in nodes:
            nodes[model_name] = {
                "kind":    model_type,
                "reqs":    0,
                "tok_in":  0,
                "tok_out": 0,
                "secs":    0.0,
            }
        node = nodes[model_name]
        node["reqs"]    += 1
        node["tok_in"]  += prompt_tok
        node["tok_out"] += compl_tok
        node["secs"]    += duration_seconds
        node["kind"]     = model_type

        _maybe_flush_snapshot(ctx)
