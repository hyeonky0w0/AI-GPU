"""041 Direct DCN의 고정 피처·시간·자산 계약."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

ID, TARGET, FOLD = "row_id", "control_success", "fold"
YEARS = (2022, 2023, 2024)
OUTER = {2022:(2019,2020,2021), 2023:(2019,2020,2021,2022), 2024:(2019,2020,2021,2022,2023)}
INNER_VALIDATION = {2022:2021, 2023:2022, 2024:2023}
RAW_FEATURES = [
 "season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before","outs_before","run_top_before","run_bot_before","score_diff_pitcher_team","base_state","home_win_expectancy","li","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
ONEHOT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand","game_dayofweek","pitcher_team_id","batter_team_id"]
NUMERIC=[c for c in RAW_FEATURES if c not in ONEHOT+["season"]]
WEIGHTS=(0.0,0.05,0.10,0.20)
def root(): return Path(__file__).resolve().parents[1]
def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def object_sha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def atomic_json(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,path)
