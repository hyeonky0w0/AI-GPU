from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_pipeline import apply_feature_policy, apply_training_window_policy, training_sample_weights
from models import fit_predict, select_nonnegative_oof_weight
from research import candidate_stage_state


class CandidateHypothesisTests(unittest.TestCase):
    def test_recent_season_weight_uses_training_seasons_only(self):
        train = pd.DataFrame({"season": [2021, 2022, 2022], "x": [1, 2, 3]})
        weights = training_sample_weights(train, "latest_season_double")
        self.assertEqual(weights.tolist(), [1.0, 2.0, 2.0])

    def test_recent_season_weight_rejects_missing_season(self):
        with self.assertRaisesRegex(ValueError, "season"):
            training_sample_weights(pd.DataFrame({"x": [1]}), "latest_season_double")

    def test_drop_player_ids_removes_only_player_ids(self):
        train = pd.DataFrame({"season": [2022], "pitcher_id": [1], "batter_id": [2], "pitcher_team_id": [3]})
        validation = train.copy()
        prepared_train, prepared_validation, dropped, _ = apply_feature_policy(train, validation, "drop_player_ids")
        self.assertEqual(dropped, ["pitcher_id", "batter_id"])
        self.assertEqual(prepared_train.columns.tolist(), ["season", "pitcher_team_id"])
        self.assertEqual(prepared_train.columns.tolist(), prepared_validation.columns.tolist())

    def test_recent_two_seasons_uses_only_latest_past_seasons(self):
        split = {
            "name": "fold_2024",
            "train_seasons": [2019, 2020, 2021, 2022, 2023],
            "validation_season": 2024,
        }
        prepared = apply_training_window_policy(split, "recent_two_seasons")
        self.assertEqual(prepared["train_seasons"], [2022, 2023])
        self.assertEqual(prepared["validation_season"], 2024)
        self.assertEqual(split["train_seasons"], [2019, 2020, 2021, 2022, 2023])

    def test_recent_two_seasons_rejects_insufficient_past_history(self):
        split = {"name": "bad", "train_seasons": [2023], "validation_season": 2024}
        with self.assertRaisesRegex(ValueError, "두 개 이상"):
            apply_training_window_policy(split, "recent_two_seasons")

    def test_catboost_seed_ensemble_records_each_seed_and_ensemble(self):
        X_train = pd.DataFrame({
            "season": [2019] * 20,
            "top_bottom": pd.Series(["T", "B"] * 10, dtype="string"),
            "game_type": pd.Series(["R"] * 20, dtype="string"),
            "base_state": pd.Series(["___", "1__"] * 10, dtype="string"),
            "numeric": np.arange(20, dtype=float),
        })
        X_val = X_train.head(6).copy()
        y_train = np.array([0, 1] * 10)
        y_val = np.array([0, 1, 0, 1, 0, 1])
        prediction, details = fit_predict("catboost", X_train, y_train, X_val, {
            "seed": 42, "seeds": [42, 2026], "iterations": 5,
            "early_stopping_rounds": 2,
            "categorical": ["top_bottom", "game_type", "base_state"],
            "max_model_seconds": 30, "min_available_memory_bytes": 0,
            "y_val": y_val,
        })
        self.assertEqual(details["seed_predictions"].shape, (6, 2))
        self.assertNotIn("models", details)
        self.assertEqual([item["seed"] for item in details["seed_results"]], [42, 2026])
        self.assertAlmostEqual(details["ensemble_brier"], float(np.mean((prediction - y_val) ** 2)))

    def test_drop_season_policy_removes_only_season(self):
        train = pd.DataFrame({"season": [2019, 2020], "x": [1.0, 2.0]})
        val = pd.DataFrame({"season": [2021], "x": [3.0]})
        prepared_train, prepared_val, dropped, _ = apply_feature_policy(train, val, "drop_season")
        self.assertEqual(dropped, ["season"])
        self.assertEqual(prepared_train.columns.tolist(), ["x"])
        self.assertEqual(prepared_train.columns.tolist(), prepared_val.columns.tolist())

    def test_drop_player_ids_removes_players_but_keeps_team_ids(self):
        train = pd.DataFrame({
            "pitcher_id": [1], "batter_id": [2],
            "pitcher_team_id": [10], "batter_team_id": [20], "x": [3.0],
        })
        val = train.copy()
        prepared_train, prepared_val, dropped, _ = apply_feature_policy(train, val, "drop_player_ids")
        self.assertEqual(dropped, ["pitcher_id", "batter_id"])
        self.assertEqual(prepared_train.columns.tolist(), ["pitcher_team_id", "batter_team_id", "x"])
        self.assertEqual(prepared_train.columns.tolist(), prepared_val.columns.tolist())

    def test_psi_policy_uses_training_seasons_only(self):
        train = pd.DataFrame({
            "season": [2019] * 100 + [2020] * 100,
            "drift": list(range(100)) + list(range(1000, 1100)),
            "stable": list(range(100)) + list(range(100)),
        })
        val_a = pd.DataFrame({"season": [2021] * 10, "drift": [0] * 10, "stable": [0] * 10})
        val_b = pd.DataFrame({"season": [2021] * 10, "drift": [99999] * 10, "stable": [99999] * 10})
        _, _, dropped_a, scores_a = apply_feature_policy(train, val_a, "training_only_psi_top")
        _, _, dropped_b, scores_b = apply_feature_policy(train, val_b, "training_only_psi_top")
        self.assertIn("drift", dropped_a)
        self.assertEqual(dropped_a, dropped_b)
        self.assertEqual(scores_a, scores_b)

    def test_oof_weight_prefers_better_prior_model(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        prediction_a = np.array([0.1, 0.9, 0.1, 0.9])
        prediction_b = 1 - prediction_a
        weight, _ = select_nonnegative_oof_weight(y, prediction_a, prediction_b)
        self.assertEqual(weight, 1.0)

    def test_benchmark_policy_never_enters_smoke_quick_rolling(self):
        candidate = {"stage_policy": "benchmark_only"}
        pending = candidate_stage_state(candidate, True, None, {}, 0.00005)
        completed = candidate_stage_state(candidate, True, None, {"benchmark": {"status": "completed"}}, 0.00005)
        self.assertEqual(pending["next_stage"], "benchmark")
        self.assertIsNone(completed["next_stage"])
        self.assertEqual(completed["stage_gate_reason"], "benchmark_already_completed")


if __name__ == "__main__":
    unittest.main()
