# Codex 연구 감독 루프

기존 `automation/run_research.ps1` 위에서 Codex가 안전한 가설 하나씩 구현하고 단위
테스트와 축소 smoke까지만 수행하도록 감독한다. 이 기능은 성능 향상이나 리더보드
점수를 보장하지 않으며 champion 승격은 사람이 검토해 수행한다.

## 실행

```powershell
powershell -ExecutionPolicy Bypass -File automation/agent/run_agent.ps1 `
  -MaxHours 6 -MaxIterations 8 -MaxFailures 3
```

기본값은 6시간, 8 iteration, 연속 실패 3회다. `MaxHours`는 최소 실행 시간이 아닌
상한이다. `needs_human`, `no_safe_hypothesis`, 실패 한도, iteration 한도 또는 시간
한도에 도달하면 즉시 종료하며 가짜 대기를 하지 않는다. 각 iteration의 Codex 호출은
정확히 한 번이고, 같은 호출 안에서만 최대 두 차례 수정하도록 prompt가 제한한다.

실행 전 `codex` CLI가 PATH에 있고 로그인·과금 정책이 준비됐는지 사람이 확인해야
한다. 감독기는 다음 unattended 옵션으로 호출한다.

- sandbox: `workspace-write`
- approval policy: `never`
- network/패키지 설치: prompt에서 금지
- sandbox 설정: `sandbox_workspace_write.network_access=false`
- structured output: `output_schema.json`

현재 환경에서는 `codex` CLI가 PATH에서 발견되지 않았으므로 실제 호출을 검증하지
못했다. CLI 버전에 따라 옵션명이 달라졌다면 실제 무인 실행 전에 `codex exec --help`로
확인해야 한다. API/구독 사용량은 모델, prompt 크기, 최대 8회의 호출에 따라 달라지며
사전에 정확히 산정할 수 없다. 6시간은 비용을 보장하는 값이 아니라 종료 상한이다.

## 안전 경계

- Codex는 `automation/`과 관련 테스트만 수정한다.
- 원본 CSV, 기존 registry 행/결과, `CHAMPION.json`은 수정하지 않는다.
- quick, rolling, research, 제출 생성은 실행하지 않는다.
- 네트워크, 패키지 설치, Git commit/push/reset/checkout/clean을 금지한다.
- validation target 또는 test 분포를 피처/가중치 선택에 쓰면 기각한다.
- 동일 config hash와 이미 시도한 가설은 다시 실행하지 않는다.
- 테스트가 실패하면 smoke를 실행하지 않는다.

CLI sandbox 설정과 prompt 모두 네트워크를 금지한다. 다만 CLI 버전별 설정 지원 여부와
OS 수준 차단은 별개이므로, 완전한 무인 실행 전 `codex exec --help`와 실행 환경의
네트워크 정책을 확인해야 한다.

## 상태와 결과

`agent_state.json`은 시작·마감 시각, 완료 iteration, 연속 실패, 시도/성공/기각 가설,
종료 사유와 마지막 run ID를 원자적으로 갱신한다. 각 `runs/<iteration_id>/`에는 prompt,
Codex JSONL, structured summary, Git diff, 테스트 출력, smoke 참조와 오류가 저장된다.

`summary.json`은 `status`, `hypothesis_id`, `config_hash`, `hypothesis`, `rationale`, `files_changed`,
`tests_passed`, `smoke_run_id`, `smoke_metrics`, `leakage_checks`, `failure_reason`,
`next_recommendation`을 반드시 포함한다.

## 테스트

다음 테스트는 실제 Codex, 모델 또는 데이터를 실행하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s automation/tests -p "test_*.py"
```

내장 mock은 성공, 프로세스 실패, timeout, malformed structured JSON을 검증하기 위한
테스트 전용 경로다. 실제 운영에서 `-MockScenario`를 사용하면 연구가 수행되지 않는다.
