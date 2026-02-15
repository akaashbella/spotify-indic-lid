"""
Thin wrapper around AI4Bharat IndicLID for South Asian (Hindi/Tamil) LID.
Uses the full two-stage model (FTR + BERT) when available; supports custom model_dir.
"""
from __future__ import annotations

import os
import sys

# Allow running from project root; IndicLID expects its own directory layout
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_DIR = os.path.join(_SCRIPT_DIR, "IndicLID", "Inference", "ai4bharat")
if os.path.isdir(_INFERENCE_DIR):
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, "IndicLID"))
    from Inference.ai4bharat.IndicLID import IndicLID as _IndicLID
else:
    _IndicLID = None

import torch
import re

# South Asian (Hindi + Tamil) language codes we care about (native + romanized)
SOUTH_ASIAN_CODES = {"hin_Deva", "hin_Latn", "tam_Tamil", "tam_Latn"}


def _softmax_logit(logit: float, all_logits: list[float]) -> float:
    """Convert single logit to probability via softmax over all logits."""
    exp = torch.exp(torch.tensor(all_logits, dtype=torch.float32))
    return (exp / exp.sum()).max().item()


class IndicLIDWrapper:
    """
    Wraps IndicLID and exposes:
    - predict(text) -> (lang_code, confidence in [0,1])
    - batch_predict(texts) -> list of (lang_code, confidence)
    - get_south_asian_confidence(text) -> max confidence among Hindi/Tamil (for mixed lyrics)
    """

    def __init__(self, model_dir: str | None = None, roman_threshold: float = 0.6):
        if _IndicLID is None:
            raise RuntimeError(
                "IndicLID not found. Clone the repo and download models. See requirements.txt."
            )
        self.model_dir = (model_dir or os.environ.get("INDICLID_MODEL_DIR")) or "models"
        self.roman_threshold = roman_threshold
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        root = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.abspath(self.model_dir) if not os.path.isabs(self.model_dir) else self.model_dir
        if not os.path.isdir(model_dir):
            model_dir = os.path.join(root, "models")
        # Set paths on the class so __init__ loads from the right place
        _IndicLID.IndicLID_FTN_path = os.path.join(model_dir, "indiclid-ftn", "model_baseline_roman.bin")
        _IndicLID.IndicLID_FTR_path = os.path.join(model_dir, "indiclid-ftr", "model_baseline_roman.bin")
        _IndicLID.IndicLID_BERT_path = os.path.join(model_dir, "indiclid-bert", "basline_nn_simple.pt")
        orig_cwd = os.getcwd()
        try:
            os.chdir(root)
            self._model = _IndicLID(
                input_threshold=0.5,
                roman_lid_threshold=self.roman_threshold,
            )
        finally:
            os.chdir(orig_cwd)

    def _result_to_confidence(self, result: tuple) -> float:
        """Convert (text, lang_code, score, model_name) to confidence in [0,1]."""
        _text, _lang, score, model_name = result
        if model_name == "IndicLID-BERT":
            # BERT returns raw logit; we don't have all logits here. Use sigmoid as proxy.
            return float(torch.sigmoid(torch.tensor(score)).item())
        return float(score)

    def predict(self, text: str) -> tuple[str, float]:
        """Single prediction: (lang_code, confidence)."""
        self._ensure_loaded()
        if not (text or text.strip()):
            return "other", 0.0
        results = self._model.batch_predict([text.strip()], batch_size=1)
        r = results[0]
        conf = self._result_to_confidence(r)
        return (r[1], conf)

    def batch_predict(self, texts: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
        """Batch prediction. Returns list of (lang_code, confidence)."""
        self._ensure_loaded()
        if not texts:
            return []
        # Clean and dedupe empty
        cleaned = [t.strip() if t else "" for t in texts]
        results = self._model.batch_predict(cleaned, batch_size=batch_size)
        out = []
        for r in results:
            if not r[0]:
                out.append(("other", 0.0))
            else:
                out.append((r[1], self._result_to_confidence(r)))
        return out

    def get_south_asian_confidence(self, lyrics: str) -> tuple[str, float]:
        """
        For mixed English/Hindi or English/Tamil lyrics, return the best South Asian
        (Hindi/Tamil) label and confidence. Runs on full text first; if top prediction
        is not South Asian, runs per-line and returns max South Asian confidence.
        """
        if not (lyrics and lyrics.strip()):
            return "other", 0.0
        self._ensure_loaded()
        text = lyrics.strip()
        # First try full text
        results = self._model.batch_predict([text], batch_size=1)
        lang, conf = results[0][1], self._result_to_confidence(results[0])
        if lang in SOUTH_ASIAN_CODES:
            return lang, conf
        # Mixed: run per-line and take max South Asian confidence
        lines = [ln.strip() for ln in re.split(r"[\n]+", text) if ln.strip()]
        if not lines:
            return "other", 0.0
        line_results = self._model.batch_predict(lines, batch_size=min(32, len(lines)))
        best_lang, best_conf = "other", 0.0
        for r in line_results:
            lang, conf = r[1], self._result_to_confidence(r)
            if lang in SOUTH_ASIAN_CODES and conf > best_conf:
                best_lang, best_conf = lang, conf
        return best_lang, best_conf

    def get_south_asian_language_confidences(self, lyrics: str) -> dict[str, float]:
        """
        Run LID per line and return max confidence per South Asian language code.
        Enables multi-label assignment: e.g. {"hin_Latn": 0.9, "tam_Latn": 0.85}
        so a song can be assigned to both Hindi and Tamil playlists.
        """
        if not (lyrics and lyrics.strip()):
            return {}
        self._ensure_loaded()
        lines = [ln.strip() for ln in re.split(r"[\n]+", lyrics.strip()) if ln.strip()]
        if not lines:
            return {}
        line_results = self._model.batch_predict(lines, batch_size=min(32, len(lines)))
        max_conf: dict[str, float] = {}
        for r in line_results:
            lang, conf = r[1], self._result_to_confidence(r)
            if lang in SOUTH_ASIAN_CODES:
                max_conf[lang] = max(max_conf.get(lang, 0.0), conf)
        return max_conf
