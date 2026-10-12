"""
Loads the two trained models (no-show and waiting time) and turns lists of
feature records into predictions. If a model file is missing or fails, the
web app keeps working and simply shows no prediction.
"""

import os

import joblib
import numpy as np
import pandas as pd

# Risk bands for the corrected no-show probability. On the test data these
# bands had real no-show rates of about 11%, 24% and 42%.
LOW_LIMIT = 0.15
HIGH_LIMIT = 0.30


def risk_level(p):
    if p is None:
        return None
    if p >= HIGH_LIMIT:
        return "high"
    if p >= LOW_LIMIT:
        return "medium"
    return "low"


def _short(name):
    return "".join(word[0] for word in name.split()).upper()


class Predictor:
    def __init__(self, model_dir):
        self.no_show_bundle = self._load(os.path.join(model_dir, "no_show_model.joblib"))
        self.wait_bundle = self._load(os.path.join(model_dir, "waiting_time_model.joblib"))

    @staticmethod
    def _load(path):
        try:
            return joblib.load(path)
        except Exception as exc:  # missing file, version mismatch, ...
            print(f"[predictor] Could not load {path}: {exc}")
            return None

    @property
    def has_no_show(self):
        return self.no_show_bundle is not None

    @property
    def has_wait(self):
        return self.wait_bundle is not None

    def version(self):
        """Short label stored in predictions.model_version (max 30 characters)."""
        parts = []
        if self.has_no_show:
            parts.append("NS-" + _short(self.no_show_bundle["model_name"]))
        if self.has_wait:
            parts.append("WT-" + _short(self.wait_bundle["model_name"]))
        return "/".join(parts)[:30] or "none"

    def no_show(self, records, base_rate):
        """No-show probability for each record (None if the model fails).

        The model was trained with class_weight='balanced', which treats
        no-shows and attendances as equally common. That inflates its raw
        probabilities, so they are shifted back to the real no-show rate.
        The corrected values match the observed rates on held-out data.
        """
        if not self.has_no_show or not records:
            return [None] * len(records)
        try:
            frame = pd.DataFrame(records)[self.no_show_bundle["features"]]
            raw = self.no_show_bundle["model"].predict_proba(frame)[:, 1]
            odds = raw / (1 - raw) * (base_rate / (1 - base_rate))
            return [float(o / (1 + o)) for o in odds]
        except Exception as exc:
            print(f"[predictor] no-show prediction failed: {exc}")
            return [None] * len(records)

    def wait(self, records):
        """Predicted waiting time in minutes for each record (None on failure)."""
        if not self.has_wait or not records:
            return [None] * len(records)
        try:
            frame = pd.DataFrame(records)[self.wait_bundle["features"]]
            pred = self.wait_bundle["model"].predict(frame)
            return [float(x) for x in np.clip(pred, 0, None)]
        except Exception as exc:
            print(f"[predictor] waiting-time prediction failed: {exc}")
            return [None] * len(records)
