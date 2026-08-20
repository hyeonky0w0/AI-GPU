# 040 사전 검증 결과

## 관찰 사실

정확한 915 OOF는 746,504행이며 `0.75 * LightGBM K10 + 0.25 * 실제 890 MLP OOF`에 `-0.03946` logit shift를 적용한다. 재현 Brier는 2022 0.243409699, 2023 0.250864418, 2024 0.248060778, 전체 0.247441025, outer 2023~2024 0.249440176이다.

## 해석/가설

현재 판정은 **CONDITIONAL**이다. full rolling 결과가 없으므로 DCN의 저차 interaction 개선 여부는 미검증이다.

## 다음 판별 실험

수동 workflow에서 seed42를 실행한다. 두 fold 비악화, 합산 0.0002 이상 개선, 비영 lambda, 비포화, MLP control 대비 우수/안정을 모두 만족할 때만 three_seed를 실행한다.
