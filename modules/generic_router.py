"""
l3mcore GenericRouter

Precision hybrid router with three-tier fallback:

  1. Embedding mode (default): encodes each keyword individually as a passage
     vector, then scores user queries against all experts using a 4-signal
     hybrid formula (max_keyword, description, mean_keyword, top3_vote).
     Final scores are softmax-normalised so the output is a real probability.

  2. Classification mode: fine-tuned BERT/RoBERTa model (AutoModelForSequenceClassification)
     for ultra-fast routing when a pre-trained classifier is available.

  3. Keyword fallback: token overlap + rapidfuzz fuzzy matching when neither
     AI method produces a score above the confidence threshold.

config.json (router section):
  model_path                  -> HuggingFace repo or local path to the router model.
  router_type                 -> 'embedding' | 'classification'
  categories_file             -> path to experts.json (must stay inside project dir).
  confidence_threshold        -> minimum score (0-1) to accept a model prediction.
  confidence_threshold_keyword -> minimum score (0-1) to accept a keyword fallback hit.
  keyword_fallback            -> true/false to enable fuzzy matching tier.
  softmax_temperature         -> sharpness of softmax normalization (default 0.15).
  scoring_weights             -> per-signal weights for hybrid scoring.
"""

import os
import re
import math
import json
import threading
from collections import OrderedDict
from pathlib import Path
from modules.logger import app_logger
from modules.utils_router import clean_text, load_classification_model

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    app_logger.warning("rapidfuzz is not installed. Keyword fallback is disabled.")

try:
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    app_logger.error("transformers/torch are not installed. GenericRouter disabled.")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Required fields per expert type used in schema validation
_EXPERT_REQUIRED_FIELDS: dict[str, list[str]] = {
    "api":    ["label", "type", "model_name"],
    "ollama": ["label", "type", "model_name"],
    "local":  ["label", "type"],
}


def _validate_expert_schema(entry: dict) -> list[str]:
    """Returns a list of validation error strings for a single expert entry."""
    errors = []
    label = entry.get("label", "").strip()
    if not label:
        errors.append("missing 'label' field")
        return errors

    expert_type = entry.get("type", "local").lower()
    required = _EXPERT_REQUIRED_FIELDS.get(expert_type, ["label", "type"])
    for field in required:
        if not entry.get(field):
            errors.append(f"expert '{label}' (type={expert_type}) missing required field '{field}'")

    # Bypass keyword validation for the fallback expert (typically label 'fallback' or id 0)
    if label != "fallback" and entry.get("id") != 0:
        keywords = entry.get("keywords", [])
        if not isinstance(keywords, list):
            errors.append(f"expert '{label}': 'keywords' must be a list")
        elif len(keywords) < 5:
            errors.append(f"expert '{label}': fewer than 5 keywords — routing quality will be poor")

    return errors


class GenericRouter:
    """
    Precision hybrid router for l3mcore.

    Routing pipeline:
      1. Embedding model (SentenceTransformer): multi-vector comparison
         using 4 signals: max_keyword, description, mean_keyword, top3_vote.
         Scores are softmax-normalised before threshold comparison.
      2. Classification model (BERT/RoBERTa): direct label classification.
      3. Keyword + fuzzy fallback (rapidfuzz): token overlap + partial ratio.

    Each expert in experts.json needs at minimum 15 keywords. The router
    builds one embedding vector per keyword at startup (not a concatenated soup),
    which allows precise single-term matching at inference time.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.enabled = False
        self._model = None
        self._tokenizer = None
        self._id2label = {}
        self.category_embeddings = {}
        self.categories = {}
        self.max_experts = 15
        self.keyword_fallback = True
        self.router_type = "classification"
        self.scoring_weights: dict = {}
        self.softmax_temperature: float = 0.15
        # LRU cache backed by OrderedDict
        self._predict_cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_max_size = 256
        self._cache_lock = threading.Lock()

        self._load_config()
        self._load_categories()
        if self.enabled and TRANSFORMERS_AVAILABLE:
            self._load_model()

    def _load_config(self):
        cfg = self.config_manager.get('router', {})

        raw_model_path = cfg.get('model_path', '')
        self.router_type = cfg.get('router_type', 'classification').lower()

        if self.router_type == 'embedding' and not os.path.exists(raw_model_path) and '/' in raw_model_path:
            self.model_path = raw_model_path
        else:
            self.model_path = os.path.abspath(raw_model_path) if raw_model_path else ''

        raw_cats = cfg.get('categories_file', 'config/experts.json')
        cats_canonical = os.path.realpath(raw_cats)
        project_root = os.path.realpath('.')
        if not cats_canonical.startswith(project_root + os.sep) and cats_canonical != project_root:
            app_logger.error(
                f"categories_file '{raw_cats}' points outside the project directory. Rejected."
            )
            self.categories_file = 'config/experts.json'
        else:
            self.categories_file = raw_cats

        self.confidence_threshold = cfg.get('confidence_threshold', 0.4)
        # Separate threshold for keyword fallback — defaults to the same value if not set
        self.confidence_threshold_keyword = cfg.get(
            'confidence_threshold_keyword', self.confidence_threshold
        )
        self.keyword_fallback = cfg.get('keyword_fallback', True)
        self.enabled = bool(raw_model_path)

        default_weights = {
            "max_keyword":  0.40,
            "description":  0.30,
            "mean_keyword": 0.20,
            "top3_vote":    0.10,
        }
        self.scoring_weights = cfg.get('scoring_weights', default_weights)
        self.softmax_temperature = cfg.get('softmax_temperature', 0.15)
        self._cache_max_size = cfg.get('cache_size', 256)

    def _load_categories(self):
        path = self.categories_file
        if not os.path.exists(path):
            app_logger.warning(f"experts.json not found at: {path}")
            return

        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            
            from modules.config_manager import deobfuscate_value
            data = deobfuscate_value(data)

            self.max_experts = data.get('max_experts', 15)
            experts_list = data.get('experts', [])

            if len(experts_list) > self.max_experts:
                app_logger.warning(
                    f"Defined {len(experts_list)} experts, exceeding design limit of "
                    f"{self.max_experts}. Loading all."
                )

            for entry in experts_list:
                # Schema validation at load time
                errors = _validate_expert_schema(entry)
                for err in errors:
                    app_logger.warning(f"experts.json validation: {err}")

                label = entry.get('label', '').strip()
                if label:
                    keywords = [k.lower() for k in entry.get('keywords', [])]
                    self.categories[label] = {
                        'config':    entry,
                        'keywords':  keywords,
                        # Pre-computed token set for keyword fallback (static data)
                        'kw_tokens': frozenset(w for kw in keywords for w in kw.split()),
                    }

            app_logger.info(
                f"GenericRouter: {len(self.categories)} categories loaded: "
                f"{list(self.categories.keys())}"
            )
        except Exception as e:
            app_logger.error(f"Error loading experts.json: {e}")

    def get_expert_config(self, label: str) -> dict | None:
        return self.categories.get(label, {}).get('config')

    def get_model_path(self, label: str) -> str | None:
        cfg = self.get_expert_config(label)
        return cfg.get('model_path') if cfg else None

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            if self.router_type == 'embedding' and self.model_path:
                pass
            else:
                app_logger.warning(
                    f"GenericRouter: model not found at '{self.model_path}'. "
                    "Keyword fallback will be used exclusively."
                )
                self.enabled = False
                return

        try:
            if self.router_type == 'embedding':
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    app_logger.error(
                        "sentence-transformers not installed. Cannot use router_type='embedding'."
                    )
                    self.enabled = False
                    return

                app_logger.info(f"Loading Semantic Embedding Router from: {self.model_path}...")
                self._model = SentenceTransformer(self.model_path)

                app_logger.info("Precomputing multi-vector representations per expert...")
                self._precompute_category_embeddings()
                app_logger.info(
                    f"Multi-vector embeddings ready for {len(self.category_embeddings)} experts."
                )
            else:
                app_logger.info(f"Loading GenericRouter Classification Model from: {self.model_path}...")
                try:
                    self._tokenizer, self._model, self._id2label = load_classification_model(
                        self.model_path
                    )
                except Exception as e:
                    app_logger.error(f"Error loading GenericRouter Model: {e}")
                    self.enabled = False

        except Exception as e:
            app_logger.error(f"Error loading GenericRouter Model: {e}")
            self.enabled = False

    def _precompute_category_embeddings(self):
        """
        Builds a multi-vector representation for every expert.

        All keyword passages are encoded in a single batched call to avoid
        repeated model forward passes, then split back per expert.

        For each expert we store:
          kw_vecs   - one embedding per keyword (max 32).
          centroid  - L2-normalised mean of all keyword vectors.
          desc_vec  - embedding of the expert description (or None).
        """
        import torch
        import torch.nn.functional as F

        # Build a flat list of all passages and record where each expert's slice is
        all_passages: list[str] = []
        expert_slices: list[tuple[str, int, int]] = []  # (label, start, end)

        for label, info in self.categories.items():
            keywords = info.get('keywords', [])[:32]
            start = len(all_passages)
            if keywords:
                all_passages.extend("passage: " + kw for kw in keywords)
            end = len(all_passages)
            expert_slices.append((label, start, end))

        # Descriptions
        desc_passages: list[tuple[str, str]] = []
        for label, info in self.categories.items():
            desc = info.get('config', {}).get('description', '')
            if desc:
                desc_passages.append((label, "passage: " + desc))

        # Single batch encode for all keywords
        all_vecs = None
        if all_passages:
            all_vecs = self._model.encode(
                all_passages, convert_to_tensor=True, show_progress_bar=False
            )

        # Single batch encode for all descriptions
        desc_vec_map: dict[str, any] = {}
        if desc_passages:
            desc_texts = [p for _, p in desc_passages]
            desc_vecs = self._model.encode(
                desc_texts, convert_to_tensor=True, show_progress_bar=False
            )
            for i, (label, _) in enumerate(desc_passages):
                desc_vec_map[label] = desc_vecs[i]

        for label, start, end in expert_slices:
            info = self.categories[label]
            keywords = info.get('keywords', [])[:32]

            if all_vecs is not None and end > start:
                kw_vecs = all_vecs[start:end]
                centroid = F.normalize(kw_vecs.mean(dim=0), dim=0)
            else:
                if label != "fallback":
                    app_logger.warning(
                        f"Expert '{label}' has no keywords — embedding quality will be poor."
                    )
                kw_vecs = None
                centroid = None

            self.category_embeddings[label] = {
                'kw_vecs':  kw_vecs,
                'centroid': centroid,
                'desc_vec': desc_vec_map.get(label),
                'n_kw':     len(keywords),
            }

    def _embed_score(self, query_vec, label_data: dict) -> float:
        w = self.scoring_weights

        kw_vecs = label_data.get('kw_vecs')
        desc_vec = label_data.get('desc_vec')

        if kw_vecs is not None:
            sims = util.cos_sim(query_vec, kw_vecs)[0].tolist()
            max_kw = max(sims)
            mean_kw = sum(sims) / len(sims)
            top3 = sorted(sims, reverse=True)[:3]
            vote = sum(1.0 for s in top3 if s >= 0.40) / max(len(top3), 1)
        else:
            max_kw = mean_kw = vote = 0.0

        desc_sim = util.cos_sim(query_vec, desc_vec).item() if desc_vec is not None else mean_kw

        return (
            w.get('max_keyword',  0.40) * max_kw  +
            w.get('description',  0.30) * desc_sim +
            w.get('mean_keyword', 0.20) * mean_kw  +
            w.get('top3_vote',    0.10) * vote
        )

    def _model_predict(self, text: str) -> tuple:
        with self._cache_lock:
            if text in self._predict_cache:
                self._predict_cache.move_to_end(text)
                return self._predict_cache[text]

        if not self._model:
            return None, 0.0

        result: tuple = (None, 0.0)
        try:
            if self.router_type == 'embedding':
                if not self.category_embeddings:
                    return None, 0.0

                query_vec = self._model.encode(
                    "query: " + text, convert_to_tensor=True, show_progress_bar=False
                )

                raw_scores: dict[str, float] = {
                    label: self._embed_score(query_vec, data)
                    for label, data in self.category_embeddings.items()
                }

                temp = self.softmax_temperature
                max_raw = max(raw_scores.values())
                exp_scores = {l: math.exp((s - max_raw) / temp) for l, s in raw_scores.items()}
                total = sum(exp_scores.values())
                norm_scores = {l: v / total for l, v in exp_scores.items()}

                best_label = max(norm_scores, key=norm_scores.__getitem__)
                best_score = norm_scores[best_label]

                app_logger.debug(
                    f"Router raw scores: { {l: f'{s:.3f}' for l, s in sorted(raw_scores.items(), key=lambda x: -x[1])[:3]} }"
                )
                result = (best_label, best_score)

            else:
                if not self._tokenizer:
                    return None, 0.0

                import torch
                inputs = self._tokenizer(
                    text, return_tensors='pt',
                    truncation=True, max_length=128, padding=True
                )
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                best_idx = int(probs.argmax())
                result = (self._id2label[best_idx], float(probs[best_idx]))

        except Exception as e:
            app_logger.error(f"Error in GenericRouter model predict: {e}")
            return None, 0.0

        with self._cache_lock:
            if len(self._predict_cache) >= self._cache_max_size:
                self._predict_cache.popitem(last=False)
            self._predict_cache[text] = result
        return result

    def _keyword_predict(self, text: str) -> tuple:
        if not self.categories:
            return None, 0.0

        text_lower = text.lower()
        text_tokens = frozenset(text_lower.split())
        scores = {}

        for label, info in self.categories.items():
            keywords = info['keywords']
            if not keywords:
                continue

            # Pre-computed token set (built once in _load_categories)
            kw_tokens = info['kw_tokens']
            overlap = len(text_tokens & kw_tokens) / max(len(kw_tokens), 1)

            fuzzy_score = 0.0
            if FUZZY_AVAILABLE:
                fuzzy_score = max(
                    fuzz.token_set_ratio(text_lower, kw) / 100.0
                    for kw in keywords
                )

            scores[label] = overlap * 0.6 + fuzzy_score * 0.4

        if not scores:
            return None, 0.0

        best_label = max(scores, key=scores.__getitem__)
        return best_label, scores[best_label]

    def predict(self, text: str) -> tuple:
        """
        Classifies the text.
        1. Tries ML model if available. Uses confidence_threshold.
        2. If below threshold, uses keyword fallback. Uses confidence_threshold_keyword.
        Returns (label, score) or ('null', score).
        """
        clean = clean_text(text)
        if not clean:
            return 'null', 0.0

        if self.enabled and self._model:
            label, score = self._model_predict(clean)
            if label and score >= self.confidence_threshold:
                if label in self.categories:
                    app_logger.info(
                        f"GenericRouter [model]: '{clean[:60]}' -> {label} ({score:.2f})"
                    )
                    return label, score
                else:
                    app_logger.warning(
                        f"GenericRouter Model predicted unknown label '{label}'. Falling back..."
                    )
            else:
                app_logger.info(
                    f"GenericRouter [model]: score {score:.2f} below threshold "
                    f"({self.confidence_threshold})."
                )

        if self.keyword_fallback:
            k_label, k_score = self._keyword_predict(clean)
            if k_label and k_score >= self.confidence_threshold_keyword:
                app_logger.info(
                    f"GenericRouter [keyword]: '{clean[:60]}' -> {k_label} ({k_score:.2f})"
                )
                return k_label, k_score

        app_logger.info(f"GenericRouter: No match found for '{clean[:60]}'")
        return 'null', 0.0

    def clear_cache(self):
        with self._cache_lock:
            self._predict_cache.clear()
        app_logger.info("GenericRouter cache cleared")

    def reload_categories(self):
        with self._cache_lock:
            app_logger.info("GenericRouter: Reloading categories and config...")
            from modules.config_manager import ConfigManager
            cfg = ConfigManager().get('router', {})
            self.confidence_threshold = cfg.get('confidence_threshold', 0.4)
            self.softmax_temperature = cfg.get('softmax_temperature', 0.15)
            self.scoring_weights = cfg.get('scoring_weights', {
                'max_keyword':  0.40,
                'description':  0.30,
                'mean_keyword': 0.20,
                'top3_vote':    0.10,
            })
            self.categories.clear()
            self.category_embeddings.clear()
            self._load_categories()
            if self.enabled and self.router_type == 'embedding':
                app_logger.info("GenericRouter: Recomputing embeddings in hot-mode...")
                self._precompute_category_embeddings()
            self._predict_cache.clear()
