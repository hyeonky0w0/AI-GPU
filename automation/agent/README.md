# Codex 연구 감독 루프

기존 `automation/run_research.ps1` 위에서 Codex가 안전한 가설 하나씩 구현하고 단위
테스트, 축소 smoke, 조건부 quick까지 수행하도록 감독한다. 이 기능은 성능 향상이나 리더보드
점수를 보장하지 않으며 champion 승격은 사람이 검토해 수행한다.

## 실행

```powershell
powershell -ExecutionPolicy Bypass -File automation/agent/run_agent.ps1 `
  -MaxHours 6 -MaxIterations 8 -MaxFailures 3
```

PATH의 npm shim 대신 실행 파일을 직접 고정하려면 절대 경로만 허용하는 `-CodexPath`를
사용한다.

```powershell
powershell -ExecutionPolicy Bypass -File automation/agent/run_agent.ps1 `
  -MaxHours 6 -MaxIterations 1 -MaxFailures 1 `
  -CodexPath C:\Users\home\AppData\Roaming\npm\codex.cmd
```

`.exe`는 절대 경로로 직접 시작하고 `.cmd`/`.bat`은 `%ComSpec% /d /s /c`를 통해
실행한다. npm의 `codex.ps1`이 먼저 검색되면 같은 디렉터리의 `codex.cmd`를 선택한다.
해석 결과와 시작 인자는 `launcher_diagnostics.jsonl`에 기록하되 민감정보 형태는
마스킹한다.

기본값은 6시간, 8 iteration, 연속 실패 3회다. `MaxHours`는 최소 실행 시간이 아닌
상한이다. `needs_human`, `no_safe_hypothesis`, iteration 한도 또는 시간
한도에 도달하면 즉시 종료하며 가짜 대기를 하지 않는다. 연속 실패 한도는
`needs_human`으로 전환된다. 각 iteration의 Codex 호출은
정확히 한 번이고, 같은 호출 안에서만 최대 두 차례 수정하도록 prompt가 제한한다.
기본적으로 남은 시간이 10분 미만이면 새 iteration을 시작하지 않고
`insufficient_time_remaining`으로 정상 종료한다. `MaxIterations`는 시간과 별도의 안전 상한이다.

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

Codex CLI 0.147.0에는 `--ask-for-approval` 옵션이 없으므로 감독기는 해당 옵션을
전달하지 않는다. unattended 정책은 지원되는 `-c approval_policy="never"` 설정으로
지정하며 `--approve-for-me`, `danger-full-access`,
`--dangerously-bypass-approvals-and-sandbox`도 사용하지 않는다.

## 안전 경계

- Codex는 `automation/`과 관련 테스트만 수정한다.
- 원본 CSV, 기존 registry 행/결과, `CHAMPION.json`은 수정하지 않는다.
- 선택한 가설의 smoke와 quick만 실행한다. rolling/full, 일반 research loop, 제출 생성은 실행하지 않는다.
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
`tests_passed`, `smoke_run_id`, `smoke_metrics`, `quick_run_id`, `quick_metrics`, `leakage_checks`, `failure_reason`,
`next_recommendation`을 반드시 포함한다.

`smoke_metrics`는 동적 object가 아니라 다음 strict 배열 형식을 사용한다.

```json
[{"name": "brier_score", "value": 0.123, "split": "validation"}]
```

실제 Codex 프로세스를 시작하기 전에 schema 전체를 재귀 검사하여 모든 object의
`additionalProperties=false`와 모든 property의 `required` 포함 여부를 검증한다.

## 테스트

다음 테스트는 실제 Codex, 모델 또는 데이터를 실행하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s automation/tests -p "test_*.py"
```

내장 mock은 성공, 프로세스 실패, timeout, malformed structured JSON을 검증하기 위한
테스트 전용 경로다. 실제 운영에서 `-MockScenario`를 사용하면 연구가 수행되지 않는다.
