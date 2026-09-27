"""
Step 2: Load the saved JSON and turn it into clean pandas DataFrames.
"""
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load_bootstrap():
    with open(DATA_DIR / "bootstrap.json") as f:
        data = json.load(f)

    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    positions = pd.DataFrame(data["element_types"])

    # team id -> team name, so player rows are readable
    team_map = teams.set_index("id")["name"].to_dict()
    players["team_name"] = players["team"].map(team_map)

    # position id -> position name (GK/DEF/MID/FWD)
    pos_map = positions.set_index("id")["singular_name_short"].to_dict()
    players["position"] = players["element_type"].map(pos_map)

    # price is stored as an int like 105 meaning £10.5m
    players["price_m"] = players["now_cost"] / 10

    # keep the columns that actually matter for a first pass
    keep_cols = [
        "id", "first_name", "second_name", "web_name",
        "team_name", "position", "price_m",
        "total_points", "points_per_game", "form",
        "minutes", "goals_scored", "assists", "clean_sheets",
        "expected_goals", "expected_assists",
        "selected_by_percent", "status",
    ]
    players_clean = players[keep_cols].copy()

    return players_clean, teams


def load_fixtures():
    with open(DATA_DIR / "fixtures.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


if __name__ == "__main__":
    players, teams = load_bootstrap()
    fixtures = load_fixtures()

    print("=== Players (top 10 by total points) ===")
    print(players.sort_values("total_points", ascending=False).head(10).to_string(index=False))

    print("\n=== Shape ===")
    print(f"Players: {players.shape}")
    print(f"Teams: {teams.shape}")
    print(f"Fixtures: {fixtures.shape}")

    # save cleaned version so we don't redo this every time
    players.to_csv(DATA_DIR / "players_clean.csv", index=False)
    print("\nSaved cleaned player data to data/players_clean.csv")
