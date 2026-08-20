# 039 — 실제 890 MLP rolling OOF

## 자산 감사 결론

`experiments/oofrog_009_mlp_blend.zip`에서 실제 리더보드 890 제출에 사용한
`mlp_snap345.pkl`, 원본 학습 코드와 결과 기록을 찾았다. checkpoint SHA-256은
`1c726f98410c000f583977d0ebb69cccdff38715da98932e505f6b24c018fe04`다.

실제 계약은 sklearn MLP `(256,128)`, seeds 42~51, epoch 3/4/5 snapshot 총 30개 평균,
37개 사용 피처 및 890 전용 전처리다. 제출 blend는 LightGBM 75% + MLP 25%다.

checkpoint와 대형 OOF는 Git에 넣지 않는다. `configs/assets.json` 구조대로 RunPod Network
Volume에 둔다. 자산 누락이나 SHA/row/target 불일치는 full training 전에 실패해야 한다.

현재 원본 위치와 준비 대상은 다음과 같다.

- 실제 890 ZIP: `experiments/oofrog_009_mlp_blend.zip` (Git ignore 유지)
- checkpoint ZIP member: `011_mlp_blend/model/mlp_snap345.pkl`
- CatBoost OOF: `experiments/007_catboost_seed_ensemble/predictions/fold_{2022,2023,2024}.npz`
- LightGBM OOF: `experiments/020_catboost_lgbm_k10_blend/predictions/fold_{2022,2023,2024}.npz`
- lightweight 비교 OOF: `experiments/037_adaptive_router_oof/outputs/oof_predictions.csv`

준비 명령은 기존 `/workspace/LG_Aimers_data/train.csv`를 건드리지 않고 그 아래에
`assets/`, `router_sources/`를 새로 만든다. 목적지에 `assets/`가 이미 있으면 덮어쓰지 않고
실패한다.

```bash
python experiments/039_real_mlp_oof/src/prepare_volume_assets.py \
  --destination /workspace/LG_Aimers_data
python experiments/039_real_mlp_oof/src/verify_assets.py \
  --asset-root /workspace/LG_Aimers_data
```

필수 layout:

```text
/workspace/LG_Aimers_data/
├── train.csv
├── assets/mlp_snap345.pkl
└── router_sources/
    ├── lightweight_oof_predictions.csv.gz
    ├── catboost/fold_2022.npz, fold_2023.npz, fold_2024.npz
    └── lightgbm/fold_2022.npz, fold_2023.npz, fold_2024.npz
```

정확한 각 파일 SHA-256은 `configs/assets.json`이 단일 계약이다. 준비 스크립트는 복사·압축
후 전 파일을 검증하며, RunPod worker도 dependency 설치와 full 학습보다 먼저 같은 검증기를
실행한다. 참고로 기존 LightGBM 2023 값은 65자리 오타였으므로 실제 원본의 64자리 SHA로
교정했다. lightweight gzip은 `mtime=0`으로 결정적으로 생성한다.

## 실행

GitHub Actions의 `LG Aimers GPU Experiment`를 수동 실행하고 `real_mlp_oof`를 선택한다.
workflow는 RunPod GPU 환경에 job을 제출하고 MLP rolling OOF, Router 비교, 로그와 보고서를
GitHub artifact로 회수한다. 정확한 sklearn MLP는 GPU 연산을 사용하지 않지만 기존 GPU/RunPod
실행 격리와 Network Volume을 재사용한다.

로컬에서는 `src/smoke_test.py`만 실행한다. 전체 학습과 Router 평가는 로컬 금지다.
