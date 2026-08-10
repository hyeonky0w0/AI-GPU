"""Supervisor 테스트 전용 Codex 대역. 실제 모델·데이터를 실행하지 않는다."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def result(status: str) -> dict:
    return {
        "status": status,
        "hypothesis_id": "mock_hypothesis",
        "config_hash": "mock_config_hash",
        "hypothesis": "mock only",
        "rationale": "supervisor state transition test",
        "files_changed": [],
        "tests_passed": status == "success",
        "smoke_run_id": "mock_smoke" if status == "success" else None,
        "smoke_metrics": {"brier": 0.25} if status == "success" else {},
        "leakage_checks": {"passed": True, "details": ["mock"]},
        "failure_reason": None if status == "success" else status,
        "next_recommendation": "none",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scenario == "timeout":
        time.sleep(10)
        return 0
    if args.scenario == "failed":
        return 7
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.scenario == "malformed":
        args.output.write_text('{"status":"success"}', encoding="utf-8")
    else:
        status = "needs_human" if args.scenario == "needs_human" else "success"
        args.output.write_text(json.dumps(result(status), ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"type": "mock", "scenario": args.scenario}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
