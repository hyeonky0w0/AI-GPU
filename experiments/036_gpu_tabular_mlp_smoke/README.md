# 036 — GPU Tabular MLP Smoke

`train.csv`의 2019~2023 행으로 간단한 PyTorch Tabular MLP를 학습하고 2024년을
시간순 validation으로 평가한다. 이 실험은 GPU/RunPod 학습 경로 점검용이며 기존
035, 007, 008, 033 실험과 OOF·제출물에 의존하지 않는다.

로컬에서는 작은 fixture를 직접 만들지 않는 한 전체 CSV 학습을 실행하지 않는다.
RunPod에서는 `gpu_mlp_smoke` execution mode로 Network Volume의 `train.csv`를 읽는다.
