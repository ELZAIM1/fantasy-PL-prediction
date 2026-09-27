"""
Step 4: Train a SEPARATE model for each of the next HORIZON gameweeks
(GW+1, GW+2, ... GW+HORIZON), all using the same "current form" features.
This gives a week-by-week predicted breakdown, not just one lump total -
useful for spotting who's peaking soon vs. who has a tough patch coming,
not just "who's good overall."

The squad optimizer still gets one number per player: the sum across all
HORIZON models' predictions.
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

DATA_DIR = Path("data")

FEATURE_COLS = [
    "avg_points_last_3",
    "avg_minutes_last_3",
    "points_last_gw",
    "was_home",
]
CATEGORICAL_COLS = ["position"]
HORIZON = 5  # must match build_training_data.py's HORIZON
TARGET_COLS = [f"target_gw_plus_{i}" for i in range(1, HORIZON + 1)]


def prepare_features(df):
    df = df.copy()
    df["was_home"] = df["was_home"].astype(int)

    # one-hot encode position for the model, but KEEP the original
    # 'position' column too - needed later for readable output and the
    # squad optimizer's position-quota constraints.
    pos_dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, pos_dummies], axis=1)
    pos_dummy_cols = list(pos_dummies.columns)

    feature_cols = FEATURE_COLS + pos_dummy_cols
    return df, feature_cols


def time_based_split(df, feature_cols, target_col, test_gws=3):
    """
    Don't shuffle-split time series data. Train on early gameweeks,
    test on the most recent ones - this mimics how you'd actually use it.

    Early in the season there may only be a handful of gameweeks with
    complete features, so test_gws is capped to always leave at least
    1 gameweek for training.
    """
    df = df.dropna(subset=feature_cols + [target_col])
    unique_gws = sorted(df["gameweek"].unique())
    n_gws = len(unique_gws)

    if n_gws < 2:
        raise ValueError(
            f"Only {n_gws} gameweek(s) have complete data for {target_col} "
            f"(need at least 2). Wait for more gameweeks to finish."
        )

    test_gws = min(test_gws, n_gws - 1)
    cutoff = unique_gws[-(test_gws + 1)]

    train = df[df["gameweek"] <= cutoff]
    test = df[df["gameweek"] > cutoff]
    return train, test


def train_one_model(train, feature_cols, target_col):
    X_train = train[feature_cols]
    y_train = train[target_col]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_one(model, test, feature_cols, target_col, label):
    X_test = test[feature_cols]
    y_test = test[target_col]
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # naive baseline: "assume recent form repeats" - just avg_points_last_3
    # with no model at all, since each target here is a single gameweek.
    baseline_mae = mean_absolute_error(y_test, test["avg_points_last_3"])

    beats = "beats" if mae < baseline_mae else "does NOT beat"
    print(f"  {label}: model MAE={mae:.3f}  baseline MAE={baseline_mae:.3f}  ({beats} baseline)")
    return mae, baseline_mae


def predict_upcoming_horizon(models, full_df, feature_cols, players_clean):
    """
    Build a features row for each player as of the LATEST gameweek played,
    then run all HORIZON models to get a predicted points value for each
    of the next HORIZON gameweeks individually.
    """
    latest = (
        full_df.sort_values("gameweek")
        .groupby("player_id")
        .tail(1)
        .copy()
    )

    latest["avg_points_last_3"] = (
        full_df.sort_values("gameweek").groupby("player_id")["total_points"]
        .rolling(3).mean().reset_index(level=0, drop=True)
    ).loc[latest.index]
    latest["points_last_gw"] = latest["total_points"]
    latest["avg_minutes_last_3"] = (
        full_df.sort_values("gameweek").groupby("player_id")["minutes"]
        .rolling(3).mean().reset_index(level=0, drop=True)
    ).loc[latest.index]
    latest["was_home"] = 0  # unknown until fixtures are mapped in; neutral default

    latest = latest.dropna(subset=feature_cols)
    X = latest[feature_cols]

    gw_pred_cols = []
    for i, model in enumerate(models, start=1):
        col_name = f"pred_gw_plus_{i}"
        latest[col_name] = model.predict(X)
        gw_pred_cols.append(col_name)

    # total across the horizon - this is what the squad optimizer maximizes
    latest["predicted_points"] = latest[gw_pred_cols].sum(axis=1)

    keep_cols = ["player_id", "web_name", "position", "team_name"] + gw_pred_cols + ["predicted_points"]
    result = latest[keep_cols]
    result = result.merge(
        players_clean[["id", "price_m"]].rename(columns={"id": "player_id"}),
        on="player_id", how="left",
    )
    return result.sort_values("predicted_points", ascending=False)


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "training_table.csv")
    players_clean = pd.read_csv(DATA_DIR / "players_clean.csv")

    df_features, feature_cols = prepare_features(df)

    models = []
    print(f"Training {HORIZON} models (one per gameweek offset)...")
    for i, target_col in enumerate(TARGET_COLS, start=1):
        train, test = time_based_split(df_features, feature_cols, target_col)
        model = train_one_model(train, feature_cols, target_col)
        models.append(model)
        evaluate_one(model, test, feature_cols, target_col, label=f"GW+{i}")

    joblib.dump(models, DATA_DIR / "points_models.pkl")
    print("\nModels saved to data/points_models.pkl")

    print(f"\n=== Predicting next {HORIZON} gameweeks (week-by-week) ===")
    predictions = predict_upcoming_horizon(models, df_features, feature_cols, players_clean)
    predictions.to_csv(DATA_DIR / "predictions.csv", index=False)

    print(f"\nTop 15 predicted scorers (total over next {HORIZON} GWs):")
    print(predictions.head(15).to_string(index=False))
