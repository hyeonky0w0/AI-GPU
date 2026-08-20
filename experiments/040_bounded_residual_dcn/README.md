# 040 Bounded Residual DCN

915 anchor의 logit에 `lambda * tanh(residual)`만 더하는 실험이다. 같은 입력·Brier loss·temporal fold로 Residual MLP와 2-layer DCN을 비교한다.

- 915 OOF: `sigmoid(logit(0.75*p_lgb + 0.25*p_mlp_890) - 0.03946)`
- outer: 2022→2023, 2022~2023→2024
- lambda 선택: outer target을 보지 않는 내부 temporal validation
- seed42가 기준을 통과해야 `three_seed` 실행 가능
- full mode는 CUDA 필수이며 로컬 full rolling은 금지한다.

GitHub Actions `LG Aimers GPU Experiment`를 수동 dispatch하고 execution mode `bounded_residual_dcn`, phase `seed42`를 선택한다. 통과 뒤 `three_seed`, 검증 ACCEPT 및 exact test prediction SHA 등록 뒤 `final_train`을 사용한다.

## Windows 로컬 CPU

`src/run_local_cpu.py`는 `verify-assets`, `smoke`, `seed42`, `evaluate`만 노출한다. CPU 출력과 checkpoint는 기존 Residual MLP 경로와 겹치지 않도록 이 실험의 `outputs/local_cpu_*` 아래를 사용한다. 기본 thread는 1, DataLoader worker는 0이며 float32 학습을 유지한다. epoch마다 시간, ETA, RSS를 출력하고 `--max-hours` 또는 Ctrl+C로 중단한 뒤 같은 명령으로 완료 epoch부터 재개할 수 있다. GPU/Actions 진입점은 계속 CUDA를 강제한다.
