# Experiment Registry

| Experiment | Hypothesis ID | Model | 상태 | 단계 | 가설 | Weighted Brier | BSS |
|---|---|---|---|---|---|---|---:|---:|
| exp_20260811_002533_baseline_logistic_smoke | baseline_logistic | logistic | completed | smoke | 로지스틱 회귀 기준선 | 0.49849165371582627 | -0.9396364286778245 |
| exp_20260811_002659_baseline_logistic_smoke | baseline_logistic | logistic | completed | smoke | 로지스틱 회귀 기준선 | 0.49849165371582627 | -0.9396364286778245 |
| exp_20260811_003921_catboost_rolling_smoke | catboost_rolling | catboost | completed | smoke | CatBoost 기준선 rolling validation | 0.25024584062107047 | 0.026290721267650752 |
| exp_20260811_003937_catboost_rolling_quick | catboost_rolling | catboost | completed | quick | CatBoost 기준선 rolling validation | 0.24811880353133575 | 0.014913122889153807 |
| exp_20260811_004235_catboost_rolling_rolling | catboost_rolling | catboost | completed | rolling | CatBoost 기준선 rolling validation | 0.2477282533187318 | 0.014137178174738918 |
| exp_20260811_010408_lightgbm_rolling_smoke | lightgbm_rolling | lightgbm | completed | smoke | LightGBM 기준선 rolling validation | 0.2572768300001622 | -0.0010669346282443648 |
| exp_20260811_012221_lightgbm_rolling_quick | lightgbm_rolling | lightgbm | completed | quick | LightGBM 기준선 rolling validation | 0.24819828180301037 | 0.01459757646007298 |
| exp_20260811_013059_lightgbm_rolling_rolling | lightgbm_rolling | lightgbm | completed | rolling | LightGBM 기준선 rolling validation | 0.24813653333176103 | 0.01251238132436927 |
| exp_20260811_013537_baseline_logistic_quick | baseline_logistic | logistic | completed | quick | 로지스틱 회귀 기준선 | 0.24962131029995627 | 0.008947836584879854 |
| exp_20260811_013621_baseline_logistic_rolling | baseline_logistic | logistic | completed | rolling | 로지스틱 회귀 기준선 | 0.2491050893699915 | 0.008657902167656939 |
| exp_20260811_013730_baseline_constant_smoke | baseline_constant | constant | completed | smoke | 학습 구간 평균 상수 기준선 | 0.25700262500000004 | 0.0 |
| exp_20260811_013806_baseline_constant_quick | baseline_constant | constant | completed | quick | 학습 구간 평균 상수 기준선 | 0.25187504706086583 | 0.0 |
| exp_20260811_013816_baseline_previous_season_smoke | baseline_previous_season | previous_season_constant | completed | smoke | 직전 시즌 평균 확률 기준선 | 0.25700262500000004 | 0.0 |
| exp_20260811_013821_baseline_previous_season_quick | baseline_previous_season | previous_season_constant | completed | quick | 직전 시즌 평균 확률 기준선 | 0.24999881336874671 | 0.007449065375919162 |
| exp_20260811_014841_lightgbm_seed_ensemble_smoke | lightgbm_seed_ensemble | lightgbm | completed | smoke | LightGBM seed ensemble | 0.2522640517034489 | -0.00031331459281291885 |
| exp_20260811_014922_catboost_seed_ensemble_smoke | catboost_seed_ensemble | catboost | completed | smoke | CatBoost seed ensemble | 0.24966649063962976 | 0.00998690416630188 |
| exp_20260811_015015_drop_season_smoke | drop_season | catboost | completed | smoke | season 피처 제거 | 0.24994738058988086 | 0.00887307936536752 |
| exp_20260811_015059_drop_drift_top_smoke | drop_drift_top | lightgbm | completed | smoke | 학습기간 PSI 상위 피처 제거 | 0.2517906176363212 | 0.0015640135390859733 |
| exp_20260811_015134_oof_weighted_blend_smoke | oof_weighted_blend | oof_blend | completed | smoke | CatBoost와 LightGBM 과거 OOF 가중 앙상블 | 0.25035778699638667 | 0.007245677482103807 |
