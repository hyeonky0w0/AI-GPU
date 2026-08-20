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

## Windows 로컬 CPU 실행

현재 확인된 로컬 환경은 Python `3.12.10`, `torch 2.13.0+cpu`, scikit-learn `1.8.0`이다.
`requirements.txt`의 범위와 충돌하지 않으므로 torch를 재설치하지 않는다. 먼저 다음처럼 기존 가상환경을
활성화한다.

```powershell
$Repo = "E:\LG_Aimers"
$DataRoot = "E:\LG_Aimers"
$AssetRoot = "E:\LG_Aimers_039_assets"
$MlpOof = "$Repo\experiments\039_real_mlp_oof\outputs\local_full\mlp_oof_predictions.csv.gz"
$Output = "$Repo\experiments\040_bounded_crossfit_residual_mlp\outputs\local_seed42"
$Resume = "$Repo\experiments\040_bounded_crossfit_residual_mlp\outputs\local_checkpoints"
$Python = "$Repo\.venv\Scripts\python.exe"
$Runner = "$Repo\experiments\040_bounded_crossfit_residual_mlp\src\run_local_cpu.py"

Set-Location $Repo
& $Python -c "import torch, sklearn; print(torch.__version__, torch.cuda.is_available(), sklearn.__version__)"
```

자산 검증:

```powershell
& $Python $Runner verify-assets `
  --data-root $DataRoot `
  --asset-root $AssetRoot `
  --mlp-oof-path $MlpOof `
  --output-dir $Output `
  --cpu-workers 1
```

CPU smoke:

```powershell
& $Python $Runner smoke --cpu-workers 1
```

Seed 42 full rolling CPU 실행:

```powershell
& $Python $Runner seed42 `
  --data-root $DataRoot `
  --asset-root $AssetRoot `
  --mlp-oof-path $MlpOof `
  --output-dir $Output `
  --checkpoint-dir $Resume `
  --cpu-workers 1 `
  --max-hours 18 `
  --memory-reserve-gb 4
```

`--cpu-workers`는 PyTorch intra-op과 BLAS thread 수를 제한하며 DataLoader worker는 항상 0이다. 시작 전에
dense transform의 보수적 peak RAM과 현재 available RAM을 계산한다. 예상 peak와 4 GiB 안전 여유를
확보하지 못하면 학습 전에 실패한다. 현재 측정된 입력 차원 추정은 121, 모델용 peak 추정은 약 2.41 GiB다.
따라서 기본 안전 여유를 포함해 최소 약 6.5 GiB available RAM을 권장한다. 메모리를 확보하지 못했다면
브라우저 등을 종료한 뒤 같은 명령을 다시 실행한다. `--memory-reserve-gb`를 낮추는 것은 시스템 정지 위험을
사용자가 명시적으로 감수할 때만 사용한다.

각 epoch 뒤에 원자 checkpoint를 먼저 저장하고 RSS, 관측 peak RSS, epoch 시간, 전체 ETA를 출력한다.
예상 총시간이 20시간 이상이면 경고한다. `--max-hours`에 도달하면 해당 epoch checkpoint 저장 후 종료 코드
130으로 안전하게 끝난다. Ctrl+C는 한 번만 누르고 Python 종료를 기다린다. 진행 중 epoch는 다시 계산될 수
있지만 마지막으로 완료·검증된 epoch/snapshot은 재사용한다. 재부팅이나 중단 뒤에는 위 seed42 명령을
그대로 다시 실행하면 된다.

학습 완료 후 평가만 다시 만드는 명령:

```powershell
& $Python $Runner evaluate `
  --evaluation-phase seed42 `
  --data-root $DataRoot `
  --asset-root $AssetRoot `
  --mlp-oof-path $MlpOof `
  --output-dir $Output `
  --checkpoint-dir $Resume `
  --cpu-workers 1 `
  --memory-reserve-gb 4
```

`three-seed`와 `final-train`도 같은 runner에 존재하지만 각각 seed42 gate와 정확한 915 test asset 계약을
통과해야 한다. 현재 작업에서는 실행하지 않는다. 일반 `train_residual.py`와 RunPod 040 adapter는
`--local-cpu`를 전달하지 않으므로 기존 CUDA 필수 검사가 유지된다.

결과는 `$Output`의 `metrics_by_fold.csv`, `metrics_overall.csv`, `cap_analysis.csv`,
`correction_analysis.csv`, `residual_oof_predictions.csv.gz`, `report.json`에 생성된다. 최종 판정은
`report.json`의 `decision`과 `seed42_extend_three_seed`를 확인한다. checkpoint는 `$Resume\seed42` 아래의
outer fold/inner/refit/seed/epoch 단위로 저장되며 `.tmp`나 완료 marker 없는 파일은 재사용하지 않는다.

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
