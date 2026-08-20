"""040 bounded residual DCN의 고정 자산·시간·피처 계약."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ID, TARGET, FOLD = "row_id", "control_success", "fold"
YEARS = (2022, 2023, 2024)
OUTER = {2023: (2022,), 2024: (2022, 2023)}
SHIFT, EPS = -0.03946, 1e-6
LAMBDAS = (0.0, 0.05, 0.10, 0.20)
RAW_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "score_diff_pitcher_team", "base_state", "home_win_expectancy", "li", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
ONEHOT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
          "game_dayofweek", "pitcher_team_id", "batter_team_id"]
NUMERIC = [c for c in RAW_FEATURES if c not in ONEHOT + ["season"]]
PREDICTION_FEATURES = ["p_915", "p_lgb", "p_cat", "p_lgb_minus_cat", "p_915_minus_lgb",
                       "p_915_minus_cat", "prediction_mean", "prediction_std", "prediction_range"]
MODEL_FEATURES = RAW_FEATURES + PREDICTION_FEATURES

def root() -> Path: return Path(__file__).resolve().parents[1]
def load_json(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()
def shifted(probability, delta=SHIFT):
    import numpy as np
    p = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1-EPS)
    return 1 / (1 + np.exp(-(np.log(p/(1-p)) + delta)))
