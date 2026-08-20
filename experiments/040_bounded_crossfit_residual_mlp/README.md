# 040 Bounded Cross-fit Residual MLP

915 확률을 anchor로 고정하고 기존 890의 37개 전처리 피처와 예측 간 차이 9개만 이용해 logit을 최대
`cap`만큼 수정한다. DCN, Transformer, Router 및 자유 가중 blend는 포함하지 않는다.

## 고정 계약

`p_915 = sigmoid(logit(0.75 * p_lgb_k10 + 0.25 * p_real_890_mlp) - 0.03946)`이다. 원본
`mlp_oof_predictions.csv.gz` SHA256은
`92eeb64f41c0b71c6d15ce9e1b339a3d92cde1c0a0841f5a823d49fb086fb316`이다. 2022/2023/2024
746,504행의 row_id, target, CatBoost seed 777, LightGBM K10 순서를 모두 교차검증한 뒤에만 학습한다.

모델은 PyTorch `256 → ReLU → 128 → ReLU → 1`, Adam, lr `1e-3`, weight decay `1e-4`, batch
512, epoch 3/4/5 raw output 평균이다. 마지막 layer는 0 초기화한다. 학습 loss는
`Brier(sigmoid(logit(p_915) + 0.15*tanh(raw)), y)`이며 평가 cap은 0/0.05/0.10/0.15다.

2023 평가는 2022만 학습한다. cap은 2022의 마지막 `game_month`를 내부 검증으로 분리해 선택한다.
2024 평가는 2022~2023을 학습하고 cap은 2022→2023 내부 temporal 검증으로 선택한다. 이후 선택된
설정으로 outer 학습연도 전체를 refit한다. 평가연도 target은 전처리, 학습, snapshot, cap 선택에 쓰지 않는다.

## Network Volume

```text
/runpod-volume/LG_Aimers_data/
├── train.csv
├── test.csv                                      # final_train만 필요
├── residual_sources/
│   ├── mlp_oof_predictions.csv.gz
│   └── baseline_915_test_predictions.csv.gz     # final_train만 필요
└── router_sources/
    ├── catboost/fold_2022.npz, fold_2023.npz, fold_2024.npz
    └── lightgbm/fold_2022.npz, fold_2023.npz, fold_2024.npz
```

`baseline_915_test_predictions.csv.gz`는 `row_id,p_915,p_lgb,p_cat` 순서다. 현재 정확한 test 자산 SHA가
확정되지 않아 `assets.json`에서 의도적으로 `null`이며, 따라서 `final_train`은 fail-closed된다. 정확한
915 test asset을 확보한 뒤 SHA를 등록해야 한다. 데이터, OOF, checkpoint는 Git에 넣지 않는다.

## 로컬 검증(학습 아님)

```powershell
.\.venv\Scripts\python.exe -m py_compile experiments/040_bounded_crossfit_residual_mlp/src/*.py
.\.venv\Scripts\python.exe experiments/040_bounded_crossfit_residual_mlp/src/smoke_test.py
.\.venv\Scripts\python.exe experiments/040_bounded_crossfit_residual_mlp/src/inspect_assets.py `
  --data-root E:\LG_Aimers `
  --asset-root E:\LG_Aimers_039_assets
```

로컬에서 `train_residual.py` full mode를 실행하지 않는다. full mode는 CUDA가 없으면 즉시 실패한다.

## GitHub Actions / RunPod

Actions의 `LG Aimers GPU Experiment`에서 다음을 선택한다.

1. `execution_mode=bounded_crossfit_residual_mlp`, `phase=seed42`
2. artifact `report.json`의 `seed42_extend_three_seed=true`일 때만 `phase=three_seed`
3. 3-seed 검증 통과 및 정확한 915 test asset SHA 등록 후에만 `phase=final_train`

필요 Secret은 `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`다. endpoint는 Network Volume이
`/runpod-volume`에 mount되고 CUDA PyTorch를 실행할 수 있어야 한다. checkpoint는 volume의
`checkpoints/040_bounded_crossfit_residual_mlp`에 원자적으로 저장되어 재실행 시 완료 단위를 재사용한다.
중단은 RunPod job cancel로 수행한다. `.tmp`는 완료 marker가 아니므로 resume에서 사용하지 않는다.

대략적인 자원은 입력 one-hot 차원에 따라 달라진다. seed42는 4회의 5-epoch 학습(두 inner 선택 + 두
outer refit)으로 수 시간, 3-seed는 그 약 3배를 예상한다. GPU 메모리는 dense 전처리 배열이 아니라
모델 batch 기준으로는 작지만, 호스트 RAM은 전체 transformed matrix 때문에 충분한 여유가 필요하다.
