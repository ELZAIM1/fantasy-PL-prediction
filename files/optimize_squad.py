"""
Step 5: Given predicted points per player, pick the optimal 15-man squad
under FPL's real constraints, then pick the best starting XI + captain
from within that squad.

Install: pip install pulp
"""
import pandas as pd
import pulp
from pathlib import Path

DATA_DIR = Path("data")

BUDGET = 100.0
SQUAD_SIZE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3

# valid starting-XI formations: (DEF, MID, FWD), always 1 GK + 11 total
VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3),
    (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3),
]


def pick_squad(predictions):
    """
    Integer linear program: maximize total predicted points subject to
    budget, squad composition, and max-3-per-club constraints.
    """
    players = predictions.to_dict("records")
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    # one binary decision variable per player: 1 = include in squad
    x = {p["player_id"]: pulp.LpVariable(f"x_{p['player_id']}", cat="Binary") for p in players}

    # objective: maximize total predicted points
    prob += pulp.lpSum(x[p["player_id"]] * p["predicted_points"] for p in players)

    # budget constraint
    prob += pulp.lpSum(x[p["player_id"]] * p["price_m"] for p in players) <= BUDGET

    # exact squad composition per position
    for pos, count in SQUAD_SIZE.items():
        prob += pulp.lpSum(
            x[p["player_id"]] for p in players if p["position"] == pos
        ) == count

    # max 3 players per real club
    clubs = set(p["team_name"] for p in players)
    for club in clubs:
        prob += pulp.lpSum(
            x[p["player_id"]] for p in players if p["team_name"] == club
        ) <= MAX_PER_CLUB

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution: {pulp.LpStatus[prob.status]}")

    selected_ids = [pid for pid, var in x.items() if var.value() == 1]
    squad = predictions[predictions["player_id"].isin(selected_ids)].copy()
    return squad


def pick_best_xi(squad):
    """
    From the 15-man squad, try every valid formation and keep whichever
    starting XI scores highest, plus the captain (2x on your top scorer).
    """
    gk = squad[squad["position"] == "GKP"].nlargest(1, "predicted_points")
    best_xi = None
    best_points = -1
    best_formation = None

    for defs, mids, fwds in VALID_FORMATIONS:
        d = squad[squad["position"] == "DEF"].nlargest(defs, "predicted_points")
        m = squad[squad["position"] == "MID"].nlargest(mids, "predicted_points")
        f = squad[squad["position"] == "FWD"].nlargest(fwds, "predicted_points")

        if len(d) < defs or len(m) < mids or len(f) < fwds:
            continue  # squad doesn't have enough players in this position

        xi = pd.concat([gk, d, m, f])
        total = xi["predicted_points"].sum()

        if total > best_points:
            best_points = total
            best_xi = xi
            best_formation = f"{defs}-{mids}-{fwds}"

    captain = best_xi.nlargest(1, "predicted_points").iloc[0]
    bench = squad[~squad["player_id"].isin(best_xi["player_id"])]

    return best_xi, best_formation, captain, bench


if __name__ == "__main__":
    predictions = pd.read_csv(DATA_DIR / "predictions.csv")

    # position codes need to match SQUAD_SIZE keys exactly
    predictions["position"] = predictions["position"].replace({"GK": "GKP"})

    print("Solving for optimal 15-man squad...")
    squad = pick_squad(predictions)

    print(f"\nSquad cost: £{squad['price_m'].sum():.1f}m / £{BUDGET}m")
    print(f"Total predicted squad points: {squad['predicted_points'].sum():.1f}")
    print("\n=== Full squad ===")
    print(squad[["web_name", "position", "team_name", "price_m", "predicted_points"]]
          .sort_values("position").to_string(index=False))

    xi, formation, captain, bench = pick_best_xi(squad)

    print(f"\n=== Best starting XI (formation {formation}) ===")
    print(xi[["web_name", "position", "team_name", "predicted_points"]]
          .sort_values("position").to_string(index=False))

    print(f"\nCaptain: {captain['web_name']} ({captain['predicted_points']:.1f} pts -> {captain['predicted_points']*2:.1f} as captain)")

    print("\n=== Bench ===")
    print(bench[["web_name", "position", "team_name", "predicted_points"]].to_string(index=False))
