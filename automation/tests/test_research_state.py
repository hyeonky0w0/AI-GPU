from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research import (
    actionable_candidate_count,
    candidate_stage_results,
    choose_candidate,
    write_run_manifest,
)


class ResearchStateTests(unittest.TestCase):
    def test_actionable_candidates_counts_only_eligible_items_with_next_stage(self):
        queue = [
            {"eligible": True, "next_stage": "quick"},
            {"eligible": True, "next_stage": None},
            {"eligible": False, "next_stage": "smoke"},
        ]
        self.assertEqual(actionable_candidate_count(queue), 1)

    def test_registry_smoke_result_does_not_become_rolling(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            result_path.write_text(json.dumps({"folds": []}), encoding="utf-8")
            rows = [{
                "hypothesis_id": "lightgbm_rolling",
                "changed_elements": "lightgbm_rolling",
                "stage": "smoke",
                "status": "completed",
                "experiment_id": "exp_smoke",
                "result_file": str(result_path),
            }]
            stages = candidate_stage_results("lightgbm_rolling", rows)
            self.assertIn("smoke", stages)
            self.assertNotIn("rolling", stages)

    def test_candidate_without_next_stage_is_not_selected(self):
        queue = [{
            "id": "done", "priority": 100, "eligible": True,
            "next_stage": None, "stage_gate_status": "completed",
        }]
        self.assertIsNone(choose_candidate(queue))

    def test_manifest_compatibility_names_are_both_written(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_run_manifest(run_dir, {"status": "completed", "selected_experiments": 0})
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
