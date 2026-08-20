# 039 — 실제 890 MLP rolling OOF 실행 준비 결과

## 변경 파일

- `.gitignore`: 039 코드·문서는 추적하되 checkpoint/대형 OOF ignore 정책은 유지
- `.github/workflows/gpu-experiment.yml`: 변경 없음(이미 `real_mlp_oof` 선택과 artifact 회수 지원)
- `configs/assets.json`: lightweight OOF SHA 추가, LightGBM 2023의 65자리 SHA 오타 교정
- `src/verify_assets.py`: Network Volume 8개 자산 존재/SHA 사전검증 추가
- `src/prepare_volume_assets.py`: 실제 원본만 복사하고 결정적 gzip 생성 후 전 자산 검증
- `src/build_mlp_oof.py`: 47→37 feature, 10 seed×3 snapshot 및 checkpoint member 계약 강화
- `src/evaluate_router.py`: CatBoost row_id와 두 MLP OOF의 fold/target/probability 검증 강화
- `README.md`: 원본 위치, 준비 명령, Network Volume layout, workflow 실행법 명시
- `runpod/lg_aimers/config/experiment_039.json`: 실제 mount `/workspace/LG_Aimers_data`와 계약 경로 명시
- `runpod/lg_aimers/src/train.py`: dependency 설치/full training 전에 SHA 검증, 로그 artifact 회수

기존 915 baseline과 MLP architecture/전처리/학습 정책은 수정하지 않았다.

## 실제 890 자산 감사

실제 자산을 찾았다. Git에서 ignore된 `experiments/oofrog_009_mlp_blend.zip` 안의
`011_mlp_blend/model/mlp_snap345.pkl`이다. 크기 53,097,996 bytes, SHA-256은
`1c726f98410c000f583977d0ebb69cccdff38715da98932e505f6b24c018fe04`로 계약과 일치한다.
같은 ZIP의 `RESULT.md`, `src/run_mlp.py`, `src/train_submit_011.py`도 확인했다. 기록에는
`snap3-5 × w=0.25`가 leaderboard 890이라고 명시돼 있다. ZIP/checkpoint/OOF는 Git에
추가하지 않는다.

Router 원본도 저장소의 ignored 자산에서 찾았다.

- CatBoost: `experiments/007_catboost_seed_ensemble/predictions/fold_{year}.npz`
- LightGBM: `experiments/020_catboost_lgbm_k10_blend/predictions/fold_{year}.npz`
- lightweight 비교: `experiments/037_adaptive_router_oof/outputs/oof_predictions.csv`

기존 `assets.json`의 LightGBM 2023 SHA는 65자리여서 유효한 SHA-256일 수 없었다. 실제 파일과
`evaluate_router.py`가 공통으로 가진 64자리 값으로 오타를 교정했다. 기존의 다른 7개 SHA는
변경하지 않았고 lightweight 결정적 gzip SHA만 새로 추가했다.

## 실제 890 코드 계약 리뷰

- 입력 47개 중 중복 8개와 선수 ID 2개를 제외한 사용 feature는 정확히 37개다.
- numeric은 median imputation → QuantileTransformer(normal, 1000, subsample 200000,
  seed 0), 별도 MissingIndicator를 사용한다. season만 StandardScaler, 8개 categorical은
  dense OneHotEncoder(handle_unknown=ignore)다. 각 fold의 train season에만 fit한다.
- sklearn MLP `(256,128)`, Adam, alpha `1e-4`, batch 512, learning rate `1e-3`을 유지한다.
- seeds 42~51 각각 epoch 1부터 연속 `partial_fit`하고 epoch 3/4/5 예측 총 30개를 산술평균한다.
- rolling fold는 2019~2021→2022, 2019~2022→2023, 2019~2023→2024다. validation/future
  season은 학습과 preprocessing fit에서 제외된다.
- `row_id`는 feature에서 제외하고 원본 순서로 OOF에 직접 보존한다. target은 `partial_fit`과
  지표/계약 검증에만 사용하며 feature에 포함되지 않는다.
- Router outer 평가도 2022→2023, 2022~2023→2024만 허용한다. base/real/lightweight OOF의
  row_id, fold, target, probability를 원본 train과 대조한다.
- 기존 제출 비교 blend `0.75 × LightGBM + 0.25 × 실제 MLP`를 유지한다.

## Network Volume layout과 준비

```text
/workspace/LG_Aimers_data/
├── train.csv
├── assets/mlp_snap345.pkl
└── router_sources/
    ├── lightweight_oof_predictions.csv.gz
    ├── catboost/fold_2022.npz, fold_2023.npz, fold_2024.npz
    └── lightgbm/fold_2022.npz, fold_2023.npz, fold_2024.npz
```

원본 자산이 있는 머신에서 실행한다. 목적지의 기존 `train.csv`는 수정하지 않으며 `assets/`가
이미 있으면 덮어쓰지 않고 실패한다.

```bash
python experiments/039_real_mlp_oof/src/prepare_volume_assets.py \
  --destination /workspace/LG_Aimers_data
python experiments/039_real_mlp_oof/src/verify_assets.py \
  --asset-root /workspace/LG_Aimers_data
```

## GitHub Actions 실행 방법

GitHub Actions → `LG Aimers GPU Experiment` → `Run workflow` → `execution_mode`에서
`real_mlp_oof` 선택 → 실행한다. workflow_dispatch가 RunPod job을 제출한다. worker는
Network Volume 파일 존재 확인 후 8개 SHA를 검사하고, 성공한 경우에만 dependency 설치와 full
rolling OOF를 시작한다. 누락/불일치는 `asset_verification.log`, `error.log`, `metrics.json`에
남고 job은 실패한다. 성공 시 MLP OOF, fold 보고서, Router 비교/weight 보고서가
`lg-aimers-real_mlp_oof-<branch>-<sha>` GitHub Actions artifact로 회수된다.

## 로컬 검증 결과

- Python compile 및 두 JSON parse: PASS
- `src/smoke_test.py`: PASS, 768행(각 fold 256), schema/확률/row_id/temporal leakage PASS
- checkpoint pipeline vs 30-member 수동 평균: max/mean absolute difference `0.0`, correlation `1.0`
- Router softmax smoke: `(300,3)`, weight sum max error `2.220446049250313e-16`
- 임시 volume 자산 준비: PASS, 8개 파일 크기/SHA 모두 계약 일치
- 빈 volume 검증: 의도대로 exit 1, 누락된 8개 전체 경로를 명확히 출력
- RunPod bundle smoke: 20,412 bytes, 13 entries, 039 verifier/config/requirements 포함, 누락 0
- 로컬 환경 sklearn 1.9.0에서 checkpoint(1.8.0) load warning이 있었다. RunPod는
  `requirements.txt`로 1.8.0을 고정하므로 실제 workflow에서는 버전을 맞춘다.
- 로컬 full training 및 Router evaluation: 실행하지 않음(정책대로 미검증)

## 위험과 다음 한 단계

실제 full runtime/메모리, 세 fold Brier와 Router 결과는 아직 미검증이다. 또한 sklearn MLP는
CUDA를 직접 사용하지 않지만 요청한 실행 격리·Network Volume·artifact 회수를 위해 GPU RunPod
job에서만 full 실행한다. 다음 한 단계는 Network Volume에 위 layout을 준비·검증한 뒤
`real_mlp_oof` workflow를 1회 수동 실행하는 것이다.
