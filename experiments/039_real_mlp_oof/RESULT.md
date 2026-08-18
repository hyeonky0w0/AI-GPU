# 039 — GPU 실행 전 구현 결과

## 관찰 사실

- 실제 890 자산을 `experiments/oofrog_009_mlp_blend.zip`에서 찾았다.
- 890 checkpoint는 `mlp_snap345.pkl`이며 SHA-256은 `1c726f98410c000f583977d0ebb69cccdff38715da98932e505f6b24c018fe04`다.
- 원본 기록은 `snap3-5 × MLP weight 0.25`가 리더보드 890을 기록했다고 명시한다.
- checkpoint 구조는 47-column 입력, 37개 사용 피처, hidden `(256,128)`, seeds 42~51,
  epoch 3/4/5 snapshot, 총 30-member soft-voting이다.
- 로컬 checkpoint inference와 constituent 수동 평균은 32행에서 max/mean absolute difference
  `0.0`, correlation `1.0`으로 일치했다.
- 768행 smoke OOF는 3개 fold, probability/schema/row_id/temporal leakage 검사를 통과했다.
- 로컬 전체 학습과 Router 평가는 실행하지 않았다.

## 해석/가설

- 실제 890 구조와 학습 계약은 재현 가능하다. 다만 full rolling weight는 GPU/RunPod artifact가
  생성돼야 확인할 수 있으므로 아직 성능 결론을 내리지 않는다.
- sklearn MLPClassifier는 CUDA를 사용하지 않는다. GPU workflow를 쓰는 이유는 기존 RunPod
  격리·Network Volume·artifact 회수 계약을 재사용하기 위해서다.

## 다음 판별 실험

- `real_mlp_oof` workflow를 실행해 실제 OOF와 고정된 038 Router 비교 artifact를 생성한다.
