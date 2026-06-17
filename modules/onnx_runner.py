import os
import re
import gc
import json
import time
import traceback
import threading
from modules.logger import app_logger

# Optional dependencies
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    app_logger.error("onnxruntime or transformers not installed. SpecificModelRunner disabled.")

# Only allow safe label characters (alphanumeric, dash, underscore)
_LABEL_RE = re.compile(r'^[A-Za-z0-9_\-]+$')


def _safe_label(label: str) -> str:
    """Raise ValueError if label contains path-unsafe characters."""
    if not _LABEL_RE.match(label):
        raise ValueError(f"Unsafe model label rejected: '{label}'")
    return label


class SpecificModelRunner:
    """
    Executes specialized ONNX models with intelligent memory management.

    Memory policy:
      - Maximum 3 models loaded simultaneously (LRU eviction).
      - Models idle for more than 5 minutes are unloaded by a background thread.

    Security:
      - Model labels validated against a safe-character regex before filesystem use.
      - model_path canonicalized and confined to models_base_path (path traversal prevention).
      - Stats file written atomically via .tmp + os.replace() to prevent corruption.

    Performance:
      - Stats flushed to disk every 30 seconds (not on every inference call).
      - ONNX sessions use separate load lock from cleanup lock to avoid contention.
      - ONNX thread count limited to 2 intra + 1 inter to prevent CPU contention
        when multiple models are loaded simultaneously.
    """

    # Task prefixes for fine-tuned T5 models
    MODEL_TASK_PREFIXES = {
        "malbec":      "translate Spanish to Bash: ",
        "syrah":       "translate Spanish to Bash: ",
        "pinot":       "translate Spanish to Bash: ",
        "grape-route": "translate Spanish to Bash: ",
        "chardonnay":  "translate Spanish to Bash: ",
    }

    # Base tokenizer for models without a specific one
    MODEL_TOKENIZER_BASE = {
        "malbec":      "t5-small",
        "syrah":       "t5-small",
        "pinot":       "t5-small",
        "grape-route": "t5-small",
        "chardonnay":  "t5-small",
    }

    MAX_MODELS_IN_MEMORY = 3
    MODEL_TTL_SECONDS = 300        # 5 minutes of inactivity
    CLEANUP_INTERVAL_SECONDS = 60  # Check every minute
    STATS_FLUSH_INTERVAL = 30      # Flush stats to disk every 30s

    def __init__(self, models_base_path="models", stats_path="data/model_stats.json"):
        self.models_base_path = os.path.realpath(models_base_path)
        self.stats_path = stats_path
        self.sessions = {}       # label -> InferenceSession or (enc, dec, dec_past)
        self.tokenizers = {}     # label -> AutoTokenizer
        self.last_access = {}    # label -> timestamp
        self.max_models = self.MAX_MODELS_IN_MEMORY

        # Create data directory once here so _save_stats never has to
        os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)

        self.stats = self._load_stats()
        self._stats_dirty = False       # True when in-memory stats differ from disk
        self._last_stats_flush = time.time()

        self._cleanup_lock = threading.Lock()  # Guards sessions/last_access cleanup
        self._load_lock = threading.Lock()     # Guards model loading (slow I/O)
        self._stop_cleanup = False

        if ONNX_AVAILABLE:
            self._cleanup_thread = threading.Thread(
                target=self._ttl_cleanup_loop,
                daemon=True,
                name="ONNX_TTL_Cleanup"
            )
            self._cleanup_thread.start()

    # ------------------------------------------------------------------ stats

    def _load_stats(self):
        try:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, 'r') as f:
                    raw = json.load(f)
                if not isinstance(raw, dict):
                    app_logger.warning("model_stats.json has unexpected format (not a dict). Resetting.")
                    return {}
                # Validate each entry: counters must be integers
                validated = {}
                for k, v in raw.items():
                    if isinstance(v, int):
                        validated[k] = v
                    else:
                        app_logger.warning(
                            f"model_stats.json: counter for '{k}' is {type(v).__name__}, "
                            "expected int. Discarding."
                        )
                return validated
        except Exception as e:
            app_logger.warning(f"Could not load model stats: {e}")
        return {}

    def _save_stats(self):
        """Atomic write: write to .tmp then replace."""
        try:
            tmp_path = self.stats_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(self.stats, f)
            os.replace(tmp_path, self.stats_path)
            self._stats_dirty = False
            self._last_stats_flush = time.time()
        except Exception as e:
            app_logger.error(f"Error saving model stats: {e}")

    def _maybe_flush_stats(self):
        """Flush stats to disk only if dirty and interval has passed."""
        if self._stats_dirty and (time.time() - self._last_stats_flush) >= self.STATS_FLUSH_INTERVAL:
            self._save_stats()

    # ------------------------------------------------------------ TTL cleanup

    def _ttl_cleanup_loop(self):
        """Background thread to unload inactive models (TTL) and flush stats."""
        while not self._stop_cleanup:
            try:
                time.sleep(self.CLEANUP_INTERVAL_SECONDS)
                self._cleanup_expired_models()
                self._maybe_flush_stats()
            except Exception as e:
                app_logger.error(f"Error in TTL cleanup loop: {e}")

    def _cleanup_expired_models(self):
        """Unloads models with no activity within the TTL."""
        with self._cleanup_lock:
            now = time.time()
            to_remove = [
                label for label, last_time in self.last_access.items()
                if now - last_time > self.MODEL_TTL_SECONDS
            ]
            for label in to_remove:
                if label in self.sessions and getattr(self.sessions[label], '_in_use', False):
                    continue
                app_logger.info(f"TTL Cleanup: Unloading '{label}' (idle {self.MODEL_TTL_SECONDS}s)")
                self.sessions.pop(label, None)
                self.tokenizers.pop(label, None)
                self.last_access.pop(label, None)
            if to_remove:
                gc.collect()

    # --------------------------------------------------------- load / memory

    def _safe_model_dir(self, label: str, model_path: str | None) -> str:
        """
        Returns a validated, canonical model directory path.
        Raises ValueError / FileNotFoundError on unsafe or missing paths.
        """
        _safe_label(label)  # SEC-8: reject unsafe label characters

        if model_path:
            # SEC-1: canonicalize and confine to models_base_path
            canonical = os.path.realpath(model_path)
            if not canonical.startswith(self.models_base_path + os.sep) and \
               canonical != self.models_base_path:
                raise ValueError(
                    f"model_path '{model_path}' escapes the models directory. Rejected."
                )
            if not os.path.isdir(canonical):
                raise FileNotFoundError(f"Model directory not found: {canonical}")
            return canonical

        # Default: models/<label>
        default = os.path.join(self.models_base_path, label)
        if not os.path.isdir(default):
            raise FileNotFoundError(f"Model directory not found: {default}")
        return default

    def _load_model_into_memory(self, label: str, model_dir: str):
        """
        Loads the model into memory.
        Uses _load_lock to allow cleanup to proceed independently.
        """
        with self._load_lock:
            # Re-check under load lock in case another thread loaded it first
            if label in self.sessions:
                with self._cleanup_lock:
                    self.last_access[label] = time.time()
                return

            # --- Tokenizer ---
            has_local_tokenizer = any(
                os.path.exists(os.path.join(model_dir, f))
                for f in ['tokenizer.json', 'spiece.model', 'tokenizer_config.json']
            )
            if has_local_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                app_logger.info(f"Tokenizer loaded from local directory: {model_dir}")
            else:
                sibling_models = ["chardonnay", "pinot", "syrah", "malbec"]
                tok_loaded = False
                for sibling in sibling_models:
                    if sibling == label:
                        continue
                    sibling_dir = os.path.join(self.models_base_path, sibling)
                    if os.path.exists(os.path.join(sibling_dir, 'tokenizer.json')):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(sibling_dir)
                            app_logger.info(f"Tokenizer for '{label}' loaded from sibling: {sibling_dir}")
                            tok_loaded = True
                            break
                        except Exception:
                            continue
                if not tok_loaded:
                    tok_base = self.MODEL_TOKENIZER_BASE.get(label, 't5-small')
                    app_logger.info(f"Downloading base tokenizer: {tok_base}")
                    tokenizer = AutoTokenizer.from_pretrained(tok_base)

            # --- ONNX files ---
            encoder_file  = os.path.join(model_dir, "encoder_model_quantized.onnx")
            decoder_file  = os.path.join(model_dir, "decoder_model_quantized.onnx")
            dec_past_file = os.path.join(model_dir, "decoder_with_past_model_quantized.onnx")
            if not os.path.exists(encoder_file):
                encoder_file  = os.path.join(model_dir, "encoder_model.onnx")
            if not os.path.exists(decoder_file):
                decoder_file  = os.path.join(model_dir, "decoder_model.onnx")
            if not os.path.exists(dec_past_file):
                dec_past_file = os.path.join(model_dir, "decoder_with_past_model.onnx")
            single_model_file = os.path.join(model_dir, "model.onnx")

            # OPT-7: limit threads to avoid CPU contention with multiple loaded models
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1

            if os.path.exists(encoder_file) and os.path.exists(decoder_file):
                app_logger.info(f"Loading Encoder-Decoder ({label})...")
                enc      = ort.InferenceSession(encoder_file,  sess_options=sess_opts)
                dec      = ort.InferenceSession(decoder_file,  sess_options=sess_opts)
                dec_past = ort.InferenceSession(dec_past_file, sess_options=sess_opts) \
                    if os.path.exists(dec_past_file) else None
                session = (enc, dec, dec_past)

            elif os.path.exists(single_model_file):
                app_logger.info(f"Loading Single Model ({label})...")
                session = ort.InferenceSession(single_model_file, sess_options=sess_opts)
            else:
                raise FileNotFoundError(f"ONNX files not found in {model_dir}")

            with self._cleanup_lock:
                self.sessions[label]    = session
                self.tokenizers[label]  = tokenizer
                self.last_access[label] = time.time()

    def _manage_memory(self, target_label: str):
        """Ensures space for the new model by applying LRU eviction."""
        with self._cleanup_lock:
            if target_label in self.sessions:
                return
            if len(self.sessions) >= self.max_models:
                lru_label = min(self.sessions.keys(), key=lambda k: self.last_access.get(k, 0))
                app_logger.info(f"LRU Eviction: Unloading '{lru_label}'.")
                del self.sessions[lru_label]
                del self.tokenizers[lru_label]
                del self.last_access[lru_label]
                gc.collect()

    # --------------------------------------------------------------- inference

    def generate_command(self, text: str, label: str, model_path: str | None = None) -> str:
        """
        Generates output from input text using the specified ONNX model.
        Returns the generated text (str).
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX libraries not available.")

        # SEC-1 + SEC-8: validate label and resolve safe model directory
        model_dir = self._safe_model_dir(label, model_path)

        # OPT-1: mark stats dirty; actual disk write happens in background
        with self._cleanup_lock:
            self.stats[label] = self.stats.get(label, 0) + 1
        self._stats_dirty = True

        # Apply task prefix if needed
        task_prefix = self.MODEL_TASK_PREFIXES.get(label, "")
        if task_prefix and not text.startswith(task_prefix):
            text = task_prefix + text

        # Manage memory and load
        try:
            self._manage_memory(label)
            self._load_model_into_memory(label, model_dir)
        except Exception as e:
            app_logger.error(f"Error loading model {label}: {e}")
            raise

        # Inference
        try:
            session   = self.sessions[label]
            tokenizer = self.tokenizers[label]

            if isinstance(session, tuple):
                # Encoder-Decoder (T5)
                encoder_session, decoder_session, dec_past_session = session

                inputs = tokenizer(
                    text, return_tensors="np",
                    padding=True, truncation=True, max_length=512
                )
                input_ids      = inputs["input_ids"].astype("int64")
                attention_mask = inputs["attention_mask"].astype("int64")

                enc_out               = encoder_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
                encoder_hidden_states = enc_out[0]

                dec1_out = decoder_session.run(
                    None,
                    {
                        "input_ids":              np.array([[0]], dtype=np.int64),
                        "encoder_hidden_states":  encoder_hidden_states,
                        "encoder_attention_mask": attention_mask,
                    }
                )
                dec_out_names = [o.name for o in decoder_session.get_outputs()]
                present_map   = {dec_out_names[i]: dec1_out[i] for i in range(1, len(dec_out_names))}

                logits        = dec1_out[0]
                next_token_id = int(logits[0, -1, :].argmax())
                generated_ids = []

                for _ in range(128):
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    generated_ids.append(next_token_id)

                    if dec_past_session is None:
                        all_ids = [0] + generated_ids
                        dec_out = decoder_session.run(
                            None,
                            {
                                "input_ids":              np.array([all_ids], dtype=np.int64),
                                "encoder_hidden_states":  encoder_hidden_states,
                                "encoder_attention_mask": attention_mask,
                            }
                        )
                        logits = dec_out[0]
                        dec_out_names2 = [o.name for o in decoder_session.get_outputs()]
                        present_map = {dec_out_names2[i]: dec_out[i] for i in range(1, len(dec_out_names2))}
                    else:
                        past_feed = {
                            "input_ids":              np.array([[next_token_id]], dtype=np.int64),
                            "encoder_hidden_states":  encoder_hidden_states,
                            "encoder_attention_mask": attention_mask,
                        }
                        for inp in dec_past_session.get_inputs():
                            if inp.name.startswith('past_key_values.'):
                                present_name = inp.name.replace('past_key_values.', 'present.')
                                if present_name in present_map:
                                    past_feed[inp.name] = present_map[present_name]

                        dec_past_out = dec_past_session.run(None, past_feed)
                        logits = dec_past_out[0]
                        dec_past_out_names = [o.name for o in dec_past_session.get_outputs()]
                        for i, name in enumerate(dec_past_out_names[1:], start=1):
                            present_map[name] = dec_past_out[i]

                    next_token_id = int(logits[0, -1, :].argmax())

                command = tokenizer.decode(generated_ids, skip_special_tokens=True)

            else:
                # Single model
                input_ids    = tokenizer(text, return_tensors="np").input_ids.astype("int64")
                output_names = [o.name for o in session.get_outputs()]
                outputs      = session.run(output_names, {session.get_inputs()[0].name: input_ids})
                logits       = outputs[0]
                predicted_ids = logits.argmax(axis=-1)
                command = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

            safe_text = text[:80].replace('\n', ' ').replace('\r', '')
            app_logger.info(f"Runner ({label}): '{safe_text}' -> '{command}'")
            return command.strip()

        except Exception as e:
            app_logger.error(f"Inference error ({label}): {e}")
            app_logger.error(traceback.format_exc())
            raise RuntimeError(f"Error executing model {label}: {e}")
