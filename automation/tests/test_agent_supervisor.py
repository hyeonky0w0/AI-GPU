from __future__ import annotations

import json
import os
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

    def make_windows_layout(self, root: Path) -> Path:
        agent_dir = root / "automation" / "agent"
        tests_dir = root / "automation" / "tests"
        agent_dir.mkdir(parents=True)
        tests_dir.mkdir(parents=True)
        for name in ["run_agent.ps1", "agent_prompt.md", "output_schema.json"]:
            shutil.copy2(ROOT / "automation" / "agent" / name, agent_dir / name)
        shutil.copy2(ROOT / "automation" / "tests" / "mock_codex.py", tests_dir / "mock_codex.py")
        return agent_dir

    def launcher_command(self, agent_dir: Path, *extra: str) -> list[str]:
        return [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(agent_dir / "run_agent.ps1"),
            "-MaxHours", "0.05", "-MaxIterations", "1", "-MaxFailures", "1",
            "-IterationTimeoutMinutes", "0.2", *extra,
        ]

    def read_launcher_log(self, agent_dir: Path) -> list[dict]:
        path = agent_dir / "launcher_diagnostics.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]

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

    def test_windows_default_agent_root_uses_script_location_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as unrelated:
            fake_root = Path(directory) / "LG_Aimers"
            agent_dir = self.make_windows_layout(fake_root)
            command = [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(agent_dir / "run_agent.ps1"),
                "-MaxHours", "0.05", "-MaxIterations", "1", "-MaxFailures", "1",
                "-MockScenario", "success",
            ]
            completed = subprocess.run(command, cwd=unrelated, text=True, capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            state = json.loads((agent_dir / "agent_state.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(state["completed_iterations"], 1)
            self.assertEqual(len(list((agent_dir / "runs").iterdir())), 1)

    def test_windows_relative_agent_root_is_relative_to_script(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as unrelated:
            fake_root = Path(directory) / "LG_Aimers"
            agent_dir = self.make_windows_layout(fake_root)
            command = [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(agent_dir / "run_agent.ps1"),
                "-MaxHours", "0.05", "-MaxIterations", "1", "-MaxFailures", "1",
                "-MockScenario", "success", "-AgentRoot", "relative-agent-state",
            ]
            completed = subprocess.run(command, cwd=unrelated, text=True, capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            expected = agent_dir / "relative-agent-state" / "agent_state.json"
            self.assertTrue(expected.exists())
            self.assertFalse((Path(unrelated) / "relative-agent-state").exists())

    def test_codex_cmd_with_spaces_uses_comspec(self):
        with tempfile.TemporaryDirectory(prefix="LG Aimers ") as directory:
            fake_root = Path(directory) / "LG Aimers"
            agent_dir = self.make_windows_layout(fake_root)
            shim = fake_root / "tools with spaces" / "codex.cmd"
            shim.parent.mkdir(parents=True)
            shim.write_text("@more >nul\n@exit /b 7\n", encoding="ascii")
            completed = subprocess.run(
                self.launcher_command(agent_dir, "-CodexPath", str(shim)),
                cwd=Path(directory), text=True, capture_output=True, timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            records = self.read_launcher_log(agent_dir)
            resolved = next(item for item in records if item["event"] == "resolved_codex")
            started = next(item for item in records if item["event"] == "process_start")
            self.assertEqual(resolved["extension"], ".cmd")
            self.assertEqual(Path(started["file_name"]), Path(os.environ["ComSpec"]))

    def test_codex_exe_uses_absolute_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory) / "LG_Aimers"
            agent_dir = self.make_windows_layout(fake_root)
            python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
            # Python을 Codex 대역으로 쓰며 첫 인자 'exec'에 해당하는 스크립트만 실행한다.
            (fake_root / "exec").write_text("import sys\nsys.stdin.read()\nraise SystemExit(7)\n", encoding="utf-8")
            completed = subprocess.run(
                self.launcher_command(agent_dir, "-CodexPath", str(python_exe)),
                cwd=directory, text=True, capture_output=True, timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            started = next(item for item in self.read_launcher_log(agent_dir) if item["event"] == "process_start")
            self.assertEqual(Path(started["file_name"]), python_exe.resolve())

    def test_powershell_npm_shim_resolves_sibling_cmd(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory) / "LG_Aimers"
            agent_dir = self.make_windows_layout(fake_root)
            shim_dir = Path(directory) / "npm shims"
            shim_dir.mkdir()
            (shim_dir / "codex.ps1").write_text("exit 99\n", encoding="utf-8")
            (shim_dir / "codex.cmd").write_text("@more >nul\n@exit /b 7\n", encoding="ascii")
            environment = os.environ.copy()
            environment["PATH"] = str(shim_dir) + os.pathsep + environment.get("PATH", "")
            completed = subprocess.run(
                self.launcher_command(agent_dir), cwd=directory, env=environment,
                text=True, capture_output=True, timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            records = self.read_launcher_log(agent_dir)
            first = next(item for item in records if item["event"] == "get_command")
            resolved = next(item for item in records if item["event"] == "resolved_codex")
            self.assertEqual(first["command_type"], "ExternalScript")
            self.assertTrue(resolved["selected_path"].lower().endswith("codex.cmd"))

    def test_missing_codex_reports_path_and_install_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory) / "LG_Aimers"
            agent_dir = self.make_windows_layout(fake_root)
            completed = subprocess.run(
                self.launcher_command(agent_dir, "-CodexCommand", "definitely_missing_codex_12345"),
                cwd=directory, text=True, capture_output=True, timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("PATH=", completed.stderr)
            self.assertIn("Get-Command codex -All", completed.stderr)
            self.assertFalse((agent_dir / "agent_state.json").exists())

    def test_invalid_nested_object_schema_is_rejected_before_process_start(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory) / "LG_Aimers"
            agent_dir = self.make_windows_layout(fake_root)
            schema_path = agent_dir / "output_schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            del schema["properties"]["smoke_metrics"]["items"]["additionalProperties"]
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            completed = subprocess.run(
                self.launcher_command(agent_dir, "-MockScenario", "success"),
                cwd=directory, text=True, capture_output=True, timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("additionalProperties=false", completed.stderr)
            self.assertFalse((agent_dir / "agent_state.json").exists())
            self.assertFalse((agent_dir / "launcher_diagnostics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
