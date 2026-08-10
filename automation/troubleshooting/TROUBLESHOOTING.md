# Troubleshooting

## [2026-08-10] Pandas StringDtype가 범주형에서 누락됨

- Run ID: 기존 002 smoke 실행(자동화 도입 전)
- Experiment ID: 002_catboost_preprocess_ablation
- 실행 모드: smoke
- 실패 단계: CatBoost Pool 생성
- 상태: 해결
- 증상: `top_bottom='T'`가 수치형 피처로 전달되어 변환 실패
- 오류 메시지 요약: `Cannot convert 'T' to float`
- 영향 범위: Pandas 3의 문자열 dtype을 object만으로 판정한 CatBoost 전처리
- 추정 원인: 문자열 dtype 판정 누락
- 확인한 실제 원인: Pandas가 해당 컬럼을 object가 아닌 StringDtype으로 읽음
- 적용한 해결: object와 `pd.StringDtype`, `is_string_dtype`를 함께 판정
- 변경 파일: `experiments/002_catboost_preprocess_ablation/src/run_ablation.py`
- 검증 방법: 네 변형 smoke 재실행
- 검증 결과: A/B/C/D 모두 학습과 predict_proba 성공
- 재발 방지: 자동화 모델 전처리도 `is_string_dtype` 사용
- 관련 로그: 기존 대화 및 `RESULT_smoke.md`
- 남은 위험: 다른 extension dtype은 모델별 별도 확인 필요

## [2026-08-10] 한글 노트북 경로의 Python 파이프 인코딩 문제

- Run ID: 자동화 도입 전 조사
- Experiment ID: 해당 없음
- 실행 모드: check
- 실패 단계: 기준 노트북 읽기
- 상태: 해결
- 증상: `1차_CatBoost.ipynb` 경로가 `1?_CatBoost.ipynb`로 전달됨
- 오류 메시지 요약: `OSError: Invalid argument`
- 영향 범위: PowerShell here-string에서 Python으로 전달한 한글 경로
- 추정 원인: 셸 파이프 경로 문자열 인코딩
- 확인한 실제 원인: Python에 전달된 경로 문자열이 손상됨
- 적용한 해결: PowerShell `Get-Content -LiteralPath -Encoding UTF8 | ConvertFrom-Json` 사용
- 변경 파일: 없음
- 검증 방법: 관련 notebook 셀 출력
- 검증 결과: CatBoost 전처리 셀 확인 성공
- 재발 방지: 자동화는 `Path(__file__)` 기반 상대 경로와 UTF-8 JSON 사용
- 관련 로그: 기존 조사 출력
- 남은 위험: 외부 콘솔의 비 UTF-8 출력 설정

## [2026-08-11 00:25] Smoke 로지스틱 회귀 수렴 경고

- Run ID: `20260811_002528_smoke`
- Experiment ID: `exp_20260811_002533_baseline_logistic_smoke`
- 실행 모드: smoke
- 실패 단계: 모델 최적화(실행 자체는 완료)
- 상태: 해결
- 증상: 세 rolling fold 모두 `lbfgs`가 50 iterations 상한에서 수렴하지 않음
- 오류 메시지 요약: `ConvergenceWarning: TOTAL NO. OF ITERATIONS REACHED LIMIT`
- 영향 범위: smoke 지표의 신뢰성; 파이프라인 성공 여부와 예측 유한성은 통과
- 추정 원인: 축소 표본에서도 47개 원본 피처와 범주형 one-hot을 최적화하기에 반복 상한이 작음
- 확인한 실제 원인: scikit-learn이 모든 fold에서 설정된 50회 상한 도달을 보고함
- 적용한 해결: smoke `max_iter`를 200으로 상향
- 변경 파일: `automation/config.json`
- 검증 방법: 동일 smoke 모드를 새 설정 해시로 재실행
- 검증 결과: `20260811_002654_smoke`에서 세 fold 모두 수렴 경고 없이 완료
- 재발 방지: 수렴 경고를 로그에서 확인하고 smoke 설정을 별도 관리
- 관련 로그: `automation/logs/20260811_002521_smoke.log`
- 남은 위험: smoke 표본은 성능 비교에 사용할 수 없음
