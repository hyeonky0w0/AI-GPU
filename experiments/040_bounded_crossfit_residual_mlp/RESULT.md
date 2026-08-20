# 040 구현 및 사전 검증 결과

## 관찰 사실

- 정확한 915 rolling OOF는 실제 890 MLP OOF와 LightGBM K10 OOF를 25:75로 섞은 뒤
  `-0.03946` logit shift를 적용한 확률이다.
- 실제 890 OOF는 746,504행, SHA256
  `92eeb64f41c0b71c6d15ce9e1b339a3d92cde1c0a0841f5a823d49fb086fb316`이다.
- 915 재현 Brier는 2022 `0.243409699`, 2023 `0.250864418`, 2024 `0.248060778`, 세 fold 전체
  `0.247441025`, 최종 outer 평가 대상 2023~2024 `0.249440176`이다.
- row_id, fold, target, CatBoost seed 777과 LightGBM 내장 사본은 원본 train 순서에서 일치했다.

## 구현 해석

Residual MLP는 anchor 모델을 업데이트하지 않으며 마지막 layer 0 초기화로 시작한다. cap 선택은 outer
평가 target을 보지 않는 내부 temporal split에서만 수행한다. cap=0은 허용 오차 내 915와 동일해야 한다.

## 판정 계약

아직 full rolling 결과를 실행하지 않았으므로 현재 판정은 **CONDITIONAL**이다. 어느 fold든
`+0.0001` 이상 악화, 전체 비개선, cap=0 불일치, 누수/정렬/유한성/포화 실패면 REJECT한다. seed42가
두 fold 비악화, 전체 `0.0002` 이상 개선, 0이 아닌 deployable cap, 비포화를 만족할 때만 3-seed로
확장한다. 3-seed는 두 fold 비악화와 전체 `0.00025` 이상 개선을 요구한다.

## 다음 판별 실험

GitHub Actions에서 seed42 phase를 한 번 수동 실행하고 artifact의 deployable 지표만 판단한다.
