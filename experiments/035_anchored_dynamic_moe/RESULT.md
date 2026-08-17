# 035 — Anchored Dynamic Mixture of Experts

## 관찰 사실

- 구현만 완료했다. 전체 모델 학습·OOF 생성·리더보드 제출은 실행하지 않았다.
- Gate는 anchor 대비 전문가별 행 단위 제곱오차 개선을 목표로 하고, 총 개입은 0.30 이하로 제한한다.
- 첫 시간 폴드는 과거 OOF가 없어 anchor만 사용한다.

## 해석/가설

- 890 Champion의 OOF 예측을 반드시 별도로 만들어야 한다. 현재 `020`의 CatBoost/LightGBM OOF를 Champion의 대체물로 쓰면 anchor 정의가 바뀌므로 이 실험의 결론이 무효가 된다.
- 구간별 편향 보정은 마지막 과거 OOF를 더 이른 OOF로 예측한 값으로만 추정하고, 검증연도에는 고정 적용한다. 과거 OOF가 한 개뿐인 경우에는 보정을 하지 않는다.

## 다음 판별 실험

Champion 및 후보 전문가의 2022~2024 시간 OOF를 같은 `row_id` 순서로 생성한 뒤, 2024 gate가 2022~2023 OOF만 사용해 Champion보다 일관되게 Brier를 낮추는지 확인한다.
