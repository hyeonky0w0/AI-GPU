# 048 Count-state + hand-matchup

046 Count-state를 보존한 상태에서 마지막 결합 후보 두 개를 평가한다.

- 후보 A: 기존 47개 + `count_state` + `hand_matchup` Joint-feature LightGBM K10
- 후보 B: 046·047 최종 OOF 확률의 고정 50:50 평균

후보 A의 피처 생성과 범주 계약은 046·047 구현을 직접 불러 재사용한다. 실행에는 다음 로컬 자산이 필요하다.

- `train.csv`
- `experiments/039_real_mlp_oof/outputs/local_full/mlp_oof_predictions.csv.gz`
- `experiments/046_count_state_cross/outputs/fold_{year}.npz`
- `experiments/047_hand_matchup_cross/outputs/fold_{year}.npz`

실행:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --smoke
.\.venv\Scripts\python.exe run_experiment.py
```

기존 실험과 OOF는 읽기만 하며 수정하지 않는다.
