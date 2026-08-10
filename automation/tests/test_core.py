from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import (
    ResearchBudget, champion_decision, config_hash, derive_stage_state, is_resume_candidate,
    past_expanding_rate, static_leakage_check, validate_predictions,
    write_failure_artifacts,
)


class CoreTests(unittest.TestCase):
    def test_lower_brier_is_improvement_direction(self):
        champion = {"recent_weighted_brier": 0.25, "fold_briers": {"2022": .25, "2023": .25, "2024": .25}}
        candidate = {"stage": "rolling", "leakage_passed": True, "recent_weighted_brier": .249, "fold_briers": {"2022": .249, "2023": .249, "2024": .249}}
        decision, _ = champion_decision(candidate, champion, {"required_improved_seasons": 2, "max_single_season_regression": .0003, "min_champion_brier_improvement": .0001})
        self.assertEqual(decision, "promote")

    def test_config_hash_blocks_equivalent_order(self):
        self.assertEqual(config_hash({"b": 2, "a": 1}), config_hash({"a": 1, "b": 2}))

    def test_future_split_is_rejected(self):
        result = static_leakage_check(["season"], "control_success", "row_id", [{"name": "bad", "train_seasons": [2024], "validation_season": 2024}])
        self.assertFalse(result["passed"])

    def test_current_target_excluded_from_expanding(self):
        values = past_expanding_rate(np.array([1, 0, 1]), np.array(["p", "p", "p"]), prior=.5, strength=2)
        self.assertAlmostEqual(values[0], .5)
        self.assertAlmostEqual(values[1], 2 / 3)

    def test_nan_prediction_fails(self):
        with self.assertRaises(ValueError):
            validate_predictions(np.array([0, 1]), np.array([.2, np.nan]))

    def test_budget_stops(self):
        budget = ResearchBudget(1, 8, time.monotonic(), completed=1)
        self.assertEqual(budget.exhausted(), (True, "max_experiments"))

    def test_single_season_improvement_not_promoted(self):
        champion = {"recent_weighted_brier": .25, "fold_briers": {"2022": .25, "2023": .25, "2024": .25}}
        candidate = {"stage": "rolling", "leakage_passed": True, "recent_weighted_brier": .2499, "fold_briers": {"2022": .2501, "2023": .2501, "2024": .249}}
        decision, _ = champion_decision(candidate, champion, {"required_improved_seasons": 2, "max_single_season_regression": .0003, "min_champion_brier_improvement": .0001})
        self.assertEqual(decision, "reject")

    def test_failure_manifest_and_error_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_failure_artifacts(path, {"mode": "check", "status": "running"}, RuntimeError("boom"))
            self.assertTrue((path / "run_manifest.json").exists())
            self.assertTrue((path / "manifest.json").exists())
            self.assertTrue((path / "error.json").exists())

    def test_resume_candidate(self):
        self.assertTrue(is_resume_candidate({"mode": "research", "status": "interrupted"}))
        self.assertFalse(is_resume_candidate({"mode": "smoke", "status": "completed"}))

    def test_completed_smoke_advances_to_quick_even_with_bad_metric(self):
        state = derive_stage_state(
            True, None,
            {"smoke": {"status": "completed", "brier": 0.9, "bss": -2.0}},
            0.00005,
        )
        self.assertEqual(state["next_stage"], "quick")
        self.assertEqual(state["stage_gate_reason"], "smoke_already_completed")

    def test_quick_metric_gate_failure_blocks_rolling(self):
        state = derive_stage_state(
            True, None,
            {"smoke": {"status": "completed"}, "quick": {"status": "completed", "brier_improvement": 0.00001}},
            0.00005,
        )
        self.assertIsNone(state["next_stage"])
        self.assertEqual(state["stage_gate_reason"], "quick_metric_gate_failed")

    def test_rolling_completion_is_distinct_from_eligibility(self):
        state = derive_stage_state(
            True, None,
            {"rolling": {"status": "completed"}},
            0.00005,
        )
        self.assertIsNone(state["next_stage"])
        self.assertEqual(state["stage_gate_reason"], "rolling_already_completed")


if __name__ == "__main__":
    unittest.main()
