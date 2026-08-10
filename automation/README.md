# LG Aimers 안전 연구 자동화

이 자동화는 리더보드 1000점 이상 경쟁권을 장기 목표로 삼되, 직접 최적화 대상은 정보 누출 없는 시간 기반 검증의 Brier 감소와 내부 BSS 증가로 제한한다. 1000점을 보장하지 않으며 로컬 개선이 리더보드 개선을 보장한다고 해석하지 않는다. 제출과 ZIP 생성은 자동화 범위 밖이다.

## 실행 환경

PowerShell 진입점은 다음 순서로 Python을 찾는다.

1. 프로젝트의 `.venv\Scripts\python.exe`
2. 활성 `VIRTUAL_ENV`
3. PATH의 `python`
4. `py -3` 실행 파일

선택한 경로는 콘솔과 manifest에 기록된다. 프로젝트 루트에서 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File automation/run_research.ps1 -Mode check
powershell -ExecutionPolicy Bypass -File automation/run_research.ps1 -Mode smoke
powershell -ExecutionPolicy Bypass -File automation/run_research.ps1 -Mode research -MaxExperiments 12 -MaxHours 8
powershell -ExecutionPolicy Bypass -File automation/run_research.ps1 -Mode resume
powershell -ExecutionPolicy Bypass -File automation/run_research.ps1 -Mode report
```

`research`는 전체 quick/rolling 데이터를 사용할 수 있고 CPU·메모리·시간 소비가 크다. 이번 구축 작업에서는 실행하지 않았다.

## 모드

- `check`: Python, 패키지, 자원, 경로, 스키마, 시즌별 행 수, 체크섬, 정적 누출 규칙을 검사한다. 모델을 만들지 않는다.
- `smoke`: 시즌별 소량 표본과 seed 42 로지스틱 회귀로 세 rolling fold의 로딩→전처리→학습→예측→지표→레지스트리 흐름을 검사한다. 성능 비교나 champion 승격에 쓰지 않는다.
- `research`: 후보 하나씩 smoke→2024 quick→필수 rolling 순으로 실행한다. quick Brier가 기준보다 설정 임계값 이상 좋아야 rolling으로 간다.
- `resume`: 중단·실패한 research manifest를 감사해 재개 대상을 기록하고, 기존 결과를 덮어쓰지 않는 새 run에서 미완료 후보부터 규칙 기반 연구를 계속한다.
- `report`: 학습 없이 레지스트리와 후보 큐, 연구 계획을 다시 만든다.
- `record-score`: 사용자가 수동 제출한 점수만 기록한다. 제출을 실행하지 않는다.

## 리더보드 점수 수동 기록

```powershell
powershell -ExecutionPolicy Bypass -File automation/run_research.ps1 `
  -Mode record-score `
  -ExperimentId exp_0012 `
  -LeaderboardScore 812.34 `
  -ScoreType public `
  -Notes "수동 제출"
```

적은 제출 결과에 calibration이나 ensemble weight를 맞추지 않는다.

## 검증과 지표

필수 rolling validation은 다음과 같다.

- 2019~2021 → 2022
- 2019~2022 → 2023
- 2019~2023 → 2024

최근 시즌 가중치는 `config.json` 한 곳에서 0.20/0.30/0.50으로 관리한다. Brier, 학습 구간 타깃 평균을 사용한 deployable constant Brier, 내부 BSS, 대회 점수 근사, AUC, LogLoss, ECE, calibration slope/intercept, 확률 범위와 편향을 저장한다. 내부 점수 공식은 저장소 `AGENTS.md`에서 확인된 `max(0, 100000 * (1 - Brier / baseline_Brier))`를 쓰되, 비공개 평가 데이터의 타깃률과 다르므로 리더보드 점수가 아니다.

핵심 선택 지표는 Brier다. AUC만 좋아지고 Brier가 나쁘면 개선으로 판정하지 않는다.

## 승격 조건

임계값은 모두 `config.json`에 있다.

- Smoke→Quick: 로딩, 필수 컬럼, 누출 검사, 피처 생성, 학습/예측, 길이, 유한성, 확률 범위, 지표와 저장이 모두 성공해야 한다.
- Quick→Rolling: 동일 2024 분할에서 기준 Brier보다 최소 `0.00005` 좋아야 하며 누출·자원 오류가 없어야 한다.
- Champion: 필수 rolling 완료, 최근 가중 Brier 최소 `0.00010` 개선, 최소 2시즌 개선, 단일 시즌 악화 `0.00030` 이하, 누출·calibration·seed·재현성 조건을 모두 확인해야 한다.
- 작은 차이는 `needs_confirmation`이다. Smoke/quick 결과는 champion이 될 수 없다.

현재 `CHAMPION.json`은 `legacy_unverified`다. CatBoost 690.69와 사용자가 제시한 LightGBM 약 740은 외부 기록이며 같은 rolling 체계에서 아직 재현되지 않았다.

## 후보 생성과 선택

`catalog/hypotheses.json`은 사람이 읽고 확장할 수 있는 24개 이상의 가설 카탈로그다. 외부 LLM이나 유료 API를 호출하지 않는다. 후보는 다음을 기준으로 규칙 기반 정렬·생략한다.

- 기존 결과 및 동일 설정 해시
- 구현된 안전 템플릿 여부
- 필수 패키지와 컬럼 가용성
- 기대 효과와 실행 비용
- 누출 위험
- 이전 실패와 그룹별 오류

새 가설은 고유 `id`, 제목, 우선순위, 비용, 누출 위험, 구현 여부, 이유를 JSON에 추가한다. 실행 코드를 구현하고 정적 누출 검사를 통과시키기 전에는 `implemented: false`로 둔다. 무제한 코드를 생성하지 않는다.

## 설정 해시와 레지스트리

해시에는 데이터 SHA-256, split, 피처 정의, 모델, 파라미터, seed, sample weight, calibration, ensemble, 단계가 포함된다. 성공한 같은 해시는 다시 실행하지 않는다. 실패한 설정은 오류를 보존하고 코드/환경 변경을 검토한 후에만 재시도한다.

- `registry/EXPERIMENT_REGISTRY.csv`: 기계 판독 레지스트리
- `registry/EXPERIMENT_REGISTRY.md`: 사람이 읽는 요약
- `registry/CHAMPION.json`: 원자적으로 보존되는 champion 상태
- `registry/HYPOTHESIS_QUEUE.json`: 실행 가능성과 생략 이유를 포함한 큐
- `registry/LEADERBOARD_SCORES.csv`: 수동 점수 기록

## 결과 구조

```text
automation/runs/<run_id>/
├─ run_manifest.json
├─ run.log
├─ research_summary.md
└─ experiments/<experiment_id>/
   ├─ hypothesis.md
   ├─ config.json
   ├─ results.csv
   ├─ result.json
   ├─ RESULT.md
   ├─ group_metrics.csv
   ├─ predictions/
   ├─ models/
   └─ error.json  # 실패 시
```

run과 experiment ID에는 시각과 모드/가설이 포함된다. 정상 JSON·manifest·champion은 임시 파일을 같은 디렉터리에 쓴 후 `os.replace`로 교체한다.

## 누출 방지

- 피처 순서는 `test.csv`에서 정하고 `row_id`와 target을 제외한다.
- train 시즌의 최대값이 validation 시즌보다 작아야 한다.
- 현재/미래 결과성 컬럼 이름을 정적으로 거부한다.
- 제공 `asof_*`는 문서상 허용되지만 원시 생성 코드가 없어 독립 재감사는 미완료로 경고한다.
- 새 target encoding, expanding, rolling은 현재 행을 `shift`한 것과 동등하게 제외하고 fold 학습 구간 내부에서만 계산해야 한다.
- 메인 데이터에는 안전한 경기 날짜·경기 ID·투구 순서 조합이 없으므로 새 행 순서 기반 rolling 후보는 현재 실행 불가다.
- `trackman_history.csv`는 설정과 코드에서 승인 전 로드하지 않는다.
- calibration과 ensemble weight는 같은 validation target에 직접 적합하지 않고 과거 OOF/calibration fold만 사용해야 한다. 해당 안전 템플릿이 완성되기 전 후보는 실행 불가다.

## 그룹 분석

가능한 컬럼에 대해 시즌, 월, 투수, 타자, 이닝, 주자 상태, 카운트, 예측 확률 구간의 행 수·Brier·실제율·예측 평균을 저장한다. 100행 미만 그룹은 불안정으로 표시한다. 현재 데이터에는 실제 구종 컬럼이 없어 구종별 분석은 생략 사유를 남긴다.

## 중단·재개와 문제 해결

사용자 중단은 manifest를 `interrupted`로 갱신하며 이미 원자적으로 저장된 결과는 유지한다. `resume`은 대상 manifest를 나열하여 재개 감사를 수행한다. 오류는 0이 아닌 종료 코드, 실패 manifest, `error.json`, traceback 로그로 남는다. 반복 가능한 해결책은 `troubleshooting/TROUBLESHOOTING.md`에 누적한다.

자동화는 원본 CSV, 기존 001/002 실험, 기존 모델을 이동·삭제·덮어쓰지 않으며 Git commit/push, 외부 API, 제출, ZIP 생성을 수행하지 않는다.
