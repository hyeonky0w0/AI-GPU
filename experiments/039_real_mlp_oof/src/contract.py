"""리더보드 890 MLP의 고정 재현 계약."""
from __future__ import annotations

import hashlib
from pathlib import Path

ID, TARGET = "row_id", "control_success"
FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
DROP_DUP = ["runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
            "run_total_before", "score_diff_home", "asof_pitcher_pitchmix_n", "away_win_expectancy"]
DROP_ID = ["pitcher_id", "batter_id"]
ONEHOT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
          "game_dayofweek", "pitcher_team_id", "batter_team_id"]
USED = [x for x in FEATURES if x not in DROP_DUP + DROP_ID]
NUMERIC = [x for x in USED if x not in ONEHOT + ["season"]]
SEEDS = list(range(42, 52)); SNAPSHOTS = [3, 4, 5]
FOLDS = {2022:[2019,2020,2021], 2023:[2019,2020,2021,2022], 2024:[2019,2020,2021,2022,2023]}
CHECKPOINT_SHA256 = "1c726f98410c000f583977d0ebb69cccdff38715da98932e505f6b24c018fe04"

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
