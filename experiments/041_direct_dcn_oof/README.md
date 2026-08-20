# 041 Direct DCN rolling OOF

기존 915를 보정하지 않고 실제 MLP와 동일한 37개 raw feature만으로 `control_success`를 직접 예측한다. Cross 2층과 256→128 ReLU deep branch를 결합해 하나의 logit을 출력하며 Brier loss로 학습한다.

각 outer 연도 직전 시즌을 inner validation으로 사용해 epoch를 고른 뒤, 해당 outer 연도 이전 전체 시즌으로 같은 epoch 수만큼 재학습한다. 전처리기는 매 stage의 train 행에만 fit한다. 모든 로컬 실행은 `src/run_local_cpu.py`를 사용한다.

```powershell
$DataRoot = "E:\data"
$AssetRoot = "E:\assets"
$MlpOof = "$AssetRoot\residual_sources\mlp_oof_predictions.csv.gz"
$Out = "$PWD\experiments\041_direct_dcn_oof\outputs\local_seed42"
$Ckpt = "$PWD\experiments\041_direct_dcn_oof\outputs\local_checkpoints"
python experiments/041_direct_dcn_oof/src/run_local_cpu.py verify-assets --data-root $DataRoot --asset-root $AssetRoot --mlp-oof-path $MlpOof --output-dir $Out
python experiments/041_direct_dcn_oof/src/run_local_cpu.py smoke --output-dir "$PWD\experiments\041_direct_dcn_oof\outputs\smoke"
python experiments/041_direct_dcn_oof/src/run_local_cpu.py seed42 --data-root $DataRoot --asset-root $AssetRoot --mlp-oof-path $MlpOof --output-dir $Out --checkpoint-dir $Ckpt --cpu-workers 1 --dataloader-workers 0 --max-hours 24
```

Ctrl+C 또는 `--max-hours` 중단 뒤 같은 seed42 명령을 다시 실행하면 atomic epoch checkpoint에서 재개한다. `evaluate` stage는 저장된 fold prediction으로 CSV/report를 재생성한다. `three-seed`는 seed42 report가 확장 조건을 통과해야 하며, `final-train`은 정확한 915 test SHA 미등록 상태에서 fail-closed한다.
