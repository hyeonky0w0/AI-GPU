# 035 — Anchored Dynamic Mixture of Experts

890점 Champion 예측 `p_anchor`를 기준으로 두고, 다른 전문가가 해당 행에서 anchor보다
나을 확률이 있을 때에만 제한적으로 개입하는 OOF 기반 메타 앙상블이다.

`p_final = p_anchor + Σ g_m(x) (p_m - p_anchor)` 이며, 항상
`Σ g_m(x) <= 0.30` 이다. Gate는 각 전문가의 `((y-p_anchor)^2 - (y-p_m)^2)`를
과거 OOF에서 회귀한다. 따라서 라벨을 본 현재 검증 행으로 gate를 적합하지 않는다.

## 실행 순서

1. Champion 및 후보 전문가의 시간 순서 OOF를 `outputs/oof/`에 준비한다.
2. `python src/train_gate.py --smoke`로 수식·누수 방지 검사를 먼저 실행한다. 이 smoke는
   NumPy만 사용하므로 최소 PyTorch RunPod 이미지에서도 동작한다.
3. 사용자가 승인한 뒤에만 실제 OOF 파일로 `python src/train_gate.py`를 실행한다.

현재 저장소에는 890 Champion의 시간 순서 MLP OOF가 없으므로, 이 실험은 새 모델 학습이나
제출물을 만들지 않는다. `020`의 CatBoost/LightGBM OOF를 Champion OOF로 잘못 대체하는 것은
금지한다.

실제 rolling gate는 `pandas`, `pyarrow`, `scikit-learn`을 요구한다. 따라서 Network Volume의
OOF가 준비되고 실행 승인이 난 뒤에는 해당 의존성을 포함한 worker 이미지를 별도로 배포한다.
