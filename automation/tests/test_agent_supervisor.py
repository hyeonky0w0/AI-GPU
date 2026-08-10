from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "automation" / "agent" / "run_agent.ps1"


class AgentSupervisorTests(unittest.TestCase):
    def run_scenario(self, scenario: str, *, failures: int = 1, timeout_minutes: float = 1):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        agent_root = Path(temporary.name) / "agent"
        command = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
            "-MaxHours", "0.05", "-MaxIterations", "1", "-MaxFailures", str(failures),
            "-IterationTimeoutMinutes", str(timeout_minutes), "-MockScenario", scenario,
            "-AgentRoot", str(agent_root),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
        state = json.loads((agent_root / "agent_state.json").read_text(encoding="utf-8-sig"))
        run_dirs = list((agent_root / "runs").iterdir())
        self.assertEqual(len(run_dirs), 1)
        return completed, state, run_dirs[0]

    def test_mock_success(self):
        completed, state, run_dir = self.run_scenario("success")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(state["successful_hypotheses"], ["mock_hypothesis"])
        self.assertEqual(state["stop_reason"], "max_iterations")
        for name in ["prompt.txt", "codex_stdout.jsonl", "summary.json", "git_diff.patch", "test_output.txt", "smoke_result_reference.json"]:
            self.assertTrue((run_dir / name).exists(), name)

    def test_mock_process_failure(self):
        completed, state, run_dir = self.run_scenario("failed")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(state["stop_reason"], "max_failures")
        self.assertTrue((run_dir / "error.json").exists())

    def test_mock_timeout(self):
        completed, state, run_dir = self.run_scenario("timeout", timeout_minutes=0.001)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(state["stop_reason"], "max_failures")
        error = json.loads((run_dir / "error.json").read_text(encoding="utf-8-sig"))
        self.assertIn("iteration_timeout", error["message"])

    def test_malformed_json_is_failure(self):
        completed, state, run_dir = self.run_scenario("malformed")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(state["stop_reason"], "max_failures")
        self.assertTrue((run_dir / "error.json").exists())


if __name__ == "__main__":
    unittest.main()
