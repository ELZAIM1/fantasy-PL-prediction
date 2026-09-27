"""
Step 1: Pull FPL data and save it locally so we have something to work with.
No auth needed - it's a public API.
"""
import requests
import json
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_bootstrap():
    """Players, teams, positions, current-season stats - the main dataset."""
    resp = requests.get(f"{BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    data = resp.json()

    with open(DATA_DIR / "bootstrap.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Players: {len(data['elements'])}")
    print(f"Teams: {len(data['teams'])}")
    print(f"Gameweeks: {len(data['events'])}")
    return data


def fetch_fixtures():
    """All fixtures for the season, including difficulty ratings."""
    resp = requests.get(f"{BASE_URL}/fixtures/")
    resp.raise_for_status()
    data = resp.json()

    with open(DATA_DIR / "fixtures.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Fixtures: {len(data)}")
    return data


if __name__ == "__main__":
    print("Fetching bootstrap-static...")
    bootstrap = fetch_bootstrap()

    print("\nFetching fixtures...")
    fixtures = fetch_fixtures()

    print("\nDone. Data saved to ./data/")
    print("\nSample player record:")
    print(json.dumps(bootstrap["elements"][0], indent=2))