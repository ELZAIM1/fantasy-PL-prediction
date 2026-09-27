"""
Step 3: Pull per-gameweek history for every player, and build a training table:
for each player-gameweek, the features are "what we knew going in" and the
label is "points scored that gameweek."
"""
import requests
import json
import time
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
BASE_URL = "https://fantasy.premierleague.com/api"

# How many gameweeks ahead to predict TOTAL points for. FPL squads are
# held across multiple gameweeks, not swapped freely each week, so
# optimizing for one week only can pick a squad that fades fast.
HORIZON = 5


def fetch_all_player_histories(player_ids):
    """
    Hits element-summary/{id}/ for every player. This is one request per
    player, so ~700 requests total - be polite about it (small delay) and
    cache the raw result so we never have to do this twice.
    """
    out_path = DATA_DIR / "player_histories.json"
    if out_path.exists():
        print("player_histories.json already exists, loading from cache.")
        with open(out_path) as f:
            return json.load(f)

    histories = {}
    for i, pid in enumerate(player_ids):
        resp = requests.get(f"{BASE_URL}/element-summary/{pid}/")
        if resp.status_code == 200:
            histories[pid] = resp.json()["history"]
        else:
            print(f"  skipped player {pid}: HTTP {resp.status_code}")

        if i % 50 == 0:
            print(f"  {i}/{len(player_ids)} players fetched...")
        time.sleep(0.05)  # be polite to the API

    with open(out_path, "w") as f:
        json.dump(histories, f)

    print(f"Done. Saved histories for {len(histories)} players.")
    return histories


def build_training_table(histories, players_clean):
    """
    Flatten every player's gameweek-by-gameweek history into one long table:
    one row per (player, gameweek) with the stats from that gameweek.
    """
    rows = []
    for pid, gw_list in histories.items():
        for gw in gw_list:
            rows.append({
                "player_id": int(pid),
                "gameweek": gw["round"],
                "opponent_team": gw["opponent_team"],
                "was_home": gw["was_home"],
                "minutes": gw["minutes"],
                "total_points": gw["total_points"],
                "goals_scored": gw["goals_scored"],
                "assists": gw["assists"],
                "clean_sheets": gw["clean_sheets"],
                "expected_goals": gw.get("expected_goals"),
                "expected_assists": gw.get("expected_assists"),
                "value": gw["value"] / 10,  # price at that gameweek
                "selected": gw["selected"],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["player_id", "gameweek"]).reset_index(drop=True)

    # merge in static info (name, position, team) for readability
    static_cols = ["id", "web_name", "position", "team_name"]
    df = df.merge(
        players_clean[static_cols].rename(columns={"id": "player_id"}),
        on="player_id", how="left",
    )

    # ROLLING FEATURES: for predicting gameweek N, we can only use info
    # from gameweeks BEFORE N - this avoids leaking the answer into the features.
    df["points_last_gw"] = df.groupby("player_id")["total_points"].shift(1)
    df["avg_points_last_3"] = (
        df.groupby("player_id")["total_points"]
        .shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    )
    df["avg_minutes_last_3"] = (
        df.groupby("player_id")["minutes"]
        .shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    )

    # FORWARD TARGETS: one column per future gameweek offset (GW+1, GW+2,
    # ... GW+HORIZON), each holding that single gameweek's actual points.
    # This lets us train a separate model per offset later and show a
    # week-by-week breakdown, rather than one opaque lump total.
    #
    # NaN on purpose (skipna not relevant here, shift() alone gives NaN)
    # when that future gameweek doesn't exist yet in the data - we drop
    # those rows at training time instead of guessing.
    for i in range(1, HORIZON + 1):
        df[f"target_gw_plus_{i}"] = df.groupby("player_id")["total_points"].shift(-i)

    return df


if __name__ == "__main__":
    players_clean = pd.read_csv(DATA_DIR / "players_clean.csv")
    player_ids = players_clean["id"].tolist()

    print(f"Fetching gameweek history for {len(player_ids)} players...")
    histories = fetch_all_player_histories(player_ids)

    print("\nBuilding training table...")
    training_df = build_training_table(histories, players_clean)

    training_df.to_csv(DATA_DIR / "training_table.csv", index=False)
    print(f"\nSaved training_table.csv with {len(training_df)} rows.")
    print("\nSample rows:")
    print(training_df.head(10).to_string(index=False))
