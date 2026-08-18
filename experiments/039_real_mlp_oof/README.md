# 039 — 실제 890 MLP rolling OOF

## 자산 감사 결론

`experiments/oofrog_009_mlp_blend.zip`에서 실제 리더보드 890 제출에 사용한
`mlp_snap345.pkl`, 원본 학습 코드와 결과 기록을 찾았다. checkpoint SHA-256은
`1c726f98410c000f583977d0ebb69cccdff38715da98932e505f6b24c018fe04`다.

실제 계약은 sklearn MLP `(256,128)`, seeds 42~51, epoch 3/4/5 snapshot 총 30개 평균,
37개 사용 피처 및 890 전용 전처리다. 제출 blend는 LightGBM 75% + MLP 25%다.

checkpoint와 대형 OOF는 Git에 넣지 않는다. `configs/assets.json` 구조대로 RunPod Network
Volume에 둔다. 자산 누락이나 SHA/row/target 불일치는 full training 전에 실패해야 한다.

## 실행

GitHub Actions의 `LG Aimers GPU Experiment`를 수동 실행하고 `real_mlp_oof`를 선택한다.
workflow는 RunPod GPU 환경에 job을 제출하고 MLP rolling OOF, Router 비교, 로그와 보고서를
GitHub artifact로 회수한다. 정확한 sklearn MLP는 GPU 연산을 사용하지 않지만 기존 GPU/RunPod
실행 격리와 Network Volume을 재사용한다.

로컬에서는 `src/smoke_test.py`만 실행한다. 전체 학습과 Router 평가는 로컬 금지다.
