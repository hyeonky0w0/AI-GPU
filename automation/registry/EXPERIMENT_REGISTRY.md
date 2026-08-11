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
| exp_20260811_025517_recent_two_seasons_smoke | recent_two_seasons | catboost | completed | smoke | 최근 2개 시즌만 학습 | 0.24978159122648924 | 0.00833049394482499 |
| exp_20260811_102133_recent_two_seasons_quick | recent_two_seasons | catboost | completed | quick | 최근 2개 시즌만 학습 | 0.2482743157686926 | 0.009331783880836886 |
| exp_20260811_102345_lightgbm_seed_ensemble_quick | lightgbm_seed_ensemble | lightgbm | completed | quick | LightGBM seed ensemble | 0.24819828180301037 | 0.01459757646007298 |
| exp_20260811_102521_catboost_seed_ensemble_quick | catboost_seed_ensemble | catboost | failed | quick | CatBoost seed ensemble |  |  |
| exp_20260811_103631_catboost_seed_ensemble_quick | catboost_seed_ensemble | catboost | completed | quick | CatBoost seed ensemble | 0.2480521360616702 | 0.015177807582788527 |
| exp_20260811_105029_drop_season_quick | drop_season | catboost | completed | quick | season 피처 제거 | 0.24918405483320188 | 0.01068383811364082 |
| exp_20260811_105446_drop_drift_top_quick | drop_drift_top | lightgbm | completed | quick | 학습기간 PSI 상위 피처 제거 | 0.24829546289276908 | 0.01421174590284735 |
| exp_20260811_105629_oof_weighted_blend_quick | oof_weighted_blend | oof_blend | completed | quick | CatBoost와 LightGBM 과거 OOF 가중 앙상블 | 0.24810175300499826 | 0.014980817273875346 |
| exp_20260811_110221_baseline_constant_benchmark | baseline_constant | constant | completed | benchmark | 학습 구간 평균 상수 기준선 | 0.2512806526774983 | 0.0 |
| exp_20260811_110253_baseline_previous_season_benchmark | baseline_previous_season | previous_season_constant | completed | benchmark | 직전 시즌 평균 확률 기준선 | 0.2500867388757978 | 0.004751316064244615 |
| exp_20260811_114611_recent_season_weight_smoke | recent_season_weight | catboost | completed | smoke | 최근 시즌 sample weight | 0.2494901478903064 | 0.010686163529215142 |
| exp_20260811_114649_recent_season_weight_quick | recent_season_weight | catboost | completed | quick | 최근 시즌 sample weight | 0.24809284487675642 | 0.015016184525795562 |
| exp_20260811_115432_drop_player_ids_smoke | drop_player_ids | catboost | completed | smoke | 원본 선수 ID 제거 | 0.2499271474191201 | 0.008953310813216597 |
| exp_20260811_115508_drop_player_ids_quick | drop_player_ids | catboost | completed | quick | 원본 선수 ID 제거 | 0.24805271024129513 | 0.015175527961874824 |
| exp_20260811_131458_lgbm_compare_recent_season_weight_smoke | lgbm_compare_recent_season_weight | lightgbm | completed | smoke | LightGBM 동일조건 최근 시즌 2배 가중 | 0.25110385127074736 | 0.005871050949272982 |
| exp_20260811_131516_lgbm_compare_recent_two_seasons_smoke | lgbm_compare_recent_two_seasons | lightgbm | completed | smoke | LightGBM 동일조건 최근 2개 시즌 학습 | 0.2514414154610146 | 0.004534622487466056 |
| exp_20260811_131534_lgbm_compare_drop_player_ids_smoke | lgbm_compare_drop_player_ids | lightgbm | completed | smoke | LightGBM 동일조건 선수 ID 제거 | 0.2519148585239992 | 0.0026602448056489703 |
| exp_20260811_131555_lgbm_compare_recent_season_weight_quick | lgbm_compare_recent_season_weight | lightgbm | completed | quick | LightGBM 동일조건 최근 시즌 2배 가중 | 0.24829708585857402 | 0.014205302367356731 |
| exp_20260811_131720_lgbm_compare_recent_two_seasons_quick | lgbm_compare_recent_two_seasons | lightgbm | failed | quick | LightGBM 동일조건 최근 2개 시즌 학습 |  |  |
| exp_20260811_144028_lgbm_compare_drop_player_ids_quick | lgbm_compare_drop_player_ids | lightgbm | completed | quick | LightGBM 동일조건 선수 ID 제거 | 0.24825904370480742 | 0.014356338185356643 |
| exp_20260811_144342_lgbm_compare_recent_two_seasons_quick | lgbm_compare_recent_two_seasons | lightgbm | failed | quick | LightGBM 동일조건 최근 2개 시즌 학습 |  |  |
| exp_20260811_144454_lgbm_compare_recent_season_weight_rolling | lgbm_compare_recent_season_weight | lightgbm | failed | rolling | LightGBM 동일조건 최근 시즌 2배 가중 |  |  |
| exp_20260811_144733_lgbm_compare_recent_two_seasons_quick | lgbm_compare_recent_two_seasons | lightgbm | completed | quick | LightGBM 동일조건 최근 2개 시즌 학습 | 0.24847801300329664 | 0.013486981331454762 |
| exp_20260811_144925_lgbm_compare_recent_season_weight_rolling | lgbm_compare_recent_season_weight | lightgbm | failed | rolling | LightGBM 동일조건 최근 시즌 2배 가중 |  |  |
| exp_20260811_145055_lgbm_compare_recent_two_seasons_rolling | lgbm_compare_recent_two_seasons | lightgbm | failed | rolling | LightGBM 동일조건 최근 2개 시즌 학습 |  |  |
