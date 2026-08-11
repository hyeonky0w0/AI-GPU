from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import validate_predictions
from models import fit_predict, prepare_lightgbm_frames


class LightGBMSafetyTests(unittest.TestCase):
    def sample_frames(self):
        train = pd.DataFrame({
            "season": [2019] * 20,
            "top_bottom": pd.Series(["T", "B"] * 10, dtype="string"),
            "game_type": pd.Series(["R"] * 19 + [None], dtype="string"),
            "base_state": pd.Series(["___", "1__"] * 10, dtype="string"),
            "numeric": np.arange(20, dtype=float),
        })
        validation = pd.DataFrame({
            "season": [2020] * 6,
            "top_bottom": pd.Series(["T", "B", "X", None, "T", "B"], dtype="string"),
            "game_type": pd.Series(["R", "P", "R", None, "R", "R"], dtype="string"),
            "base_state": pd.Series(["___", "1__", "123", None, "___", "1__"], dtype="string"),
            "numeric": np.arange(6, dtype=float),
        })
        return train, validation

    def test_string_dtype_missing_and_unseen_are_safe(self):
        train, validation = self.sample_frames()
        prepared_train, prepared_val, categorical, failures = prepare_lightgbm_frames(
            train, validation, ["top_bottom", "game_type", "base_state"]
        )
        self.assertEqual(categorical, ["top_bottom", "game_type", "base_state"])
        self.assertFalse(prepared_train[categorical].isna().any().any())
        self.assertFalse(prepared_val[categorical].isna().any().any())
        self.assertEqual(failures, {})
        self.assertIn("__UNSEEN__", prepared_val["top_bottom"].astype(str).tolist())

    def test_two_seeds_and_ensemble_are_recorded(self):
        train, validation = self.sample_frames()
        y_train = np.array([0, 1] * 10)
        y_val = np.array([0, 1, 0, 1, 0, 1])
        prediction, details = fit_predict("lightgbm", train, y_train, validation, {
            "seed": 42,
            "seeds": [42, 2026],
            "n_estimators": 5,
            "learning_rate": 0.05,
            "num_leaves": 7,
            "min_child_samples": 2,
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "subsample_freq": 0,
            "early_stopping_rounds": 2,
            "categorical": ["top_bottom", "game_type", "base_state"],
            "n_jobs": 1,
            "max_model_seconds": 30,
            "min_available_memory_bytes": 0,
            "y_val": y_val,
        })
        validate_predictions(y_val, prediction)
        self.assertEqual(details["seed_predictions"].shape, (6, 2))
        self.assertEqual([item["seed"] for item in details["seed_results"]], [42, 2026])
        self.assertAlmostEqual(details["ensemble_brier"], float(np.mean((prediction - y_val) ** 2)))

    def test_lightgbm_accepts_training_sample_weight(self):
        train, validation = self.sample_frames()
        y_train = np.array([0, 1] * 10)
        prediction, _ = fit_predict("lightgbm", train, y_train, validation, {
            "seed": 42, "seeds": [42], "n_estimators": 3, "min_child_samples": 2,
            "early_stopping_rounds": 2, "categorical": ["top_bottom", "game_type", "base_state"],
            "n_jobs": 1, "max_model_seconds": 30, "min_available_memory_bytes": 0,
            "sample_weight": np.linspace(1.0, 2.0, len(y_train)),
            "y_val": np.array([0, 1, 0, 1, 0, 1]),
        })
        self.assertEqual(len(prediction), len(validation))


if __name__ == "__main__":
    unittest.main()
