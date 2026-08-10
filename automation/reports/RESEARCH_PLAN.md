# 향후 연구 계획

후보는 구현 eligibility와 stage 진행 상태를 분리한다.
`MaxHours`는 실행 상한이며 최소 실행 시간이 아니다. actionable candidate가 없으면 즉시 정상 종료한다.
가짜 대기나 동일 실험 반복으로 시간을 채우지 않는다.

| 우선순위 | 후보 | Eligible | Next stage | Gate | 사유 |
|---:|---|---|---|---|---|
| 100 | LightGBM 기준선 rolling validation | 예 | - | completed | rolling_already_completed |
| 95 | CatBoost 기준선 rolling validation | 예 | - | completed | rolling_already_completed |
| 94 | LightGBM seed ensemble | 예 | quick | passed | smoke_already_completed |
| 93 | CatBoost seed ensemble | 예 | quick | passed | smoke_already_completed |
| 90 | 최근 시즌 sample weight | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 88 | LightGBM 확률 보정 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 86 | CatBoost 확률 보정 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 84 | 최근 2개 시즌만 학습 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 82 | CatBoost와 LightGBM 단순 평균 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 80 | 원본 선수 ID 제거 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 78 | CatBoost 범주형 A/B/C/D 재검증 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 76 | 확률 clipping 및 global prior shrinkage | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 72 | season 피처 제거 | 예 | quick | passed | smoke_already_completed |
| 70 | 학습기간 PSI 상위 피처 제거 | 예 | quick | passed | smoke_already_completed |
| 69 | CatBoost와 LightGBM 과거 OOF 가중 앙상블 | 예 | quick | passed | smoke_already_completed |
| 65 | 투수 expanding 성공률과 표본 수 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 64 | 타자 expanding 성공률과 표본 수 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 62 | 앙상블 후 calibration | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 60 | ID frequency encoding | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 58 | 누출 없는 smoothed target encoding | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 50 | 로지스틱 회귀 기준선 | 예 | - | completed | rolling_already_completed |
| 45 | 투수×구종 smoothing 통계 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 44 | 타자×구종 smoothing 통계 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 40 | 최근 N개 투구 rolling 통계 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 39 | 시간 감쇠 EWM 통계 | 아니오 | - | blocked | 안전 실행 템플릿 미구현 |
| 35 | 학습 구간 평균 상수 기준선 | 예 | benchmark | pending | benchmark_not_started |
| 34 | 직전 시즌 평균 확률 기준선 | 예 | benchmark | pending | benchmark_not_started |
