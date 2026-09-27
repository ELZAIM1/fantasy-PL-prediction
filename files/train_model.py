"""
Step 4: Train a model to predict next-gameweek points, then use it to
predict points for the UPCOMING gameweek for every player.
"""
import pandas as pd
import numpy as np
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
TARGET_COL = "total_points"


def prepare_features(df):
    df = df.copy()
    df["was_home"] = df["was_home"].astype(int)

    # one-hot encode position (GK/DEF/MID/FWD)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix="pos")
    pos_dummy_cols = [c for c in df.columns if c.startswith("pos_")]

    feature_cols = FEATURE_COLS + pos_dummy_cols
    return df, feature_cols


def time_based_split(df, feature_cols, test_gws=3):
    """
    Don't shuffle-split time series data. Train on early gameweeks,
    test on the most recent ones - this mimics how you'd actually use it.
    """
    df = df.dropna(subset=feature_cols + [TARGET_COL])
    max_gw = df["gameweek"].max()
    cutoff = max_gw - test_gws

    train = df[df["gameweek"] <= cutoff]
    test = df[df["gameweek"] > cutoff]
    return train, test


def train_model(train, feature_cols):
    X_train = train[feature_cols]
    y_train = train[TARGET_COL]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, test, feature_cols):
    X_test = test[feature_cols]
    y_test = test[TARGET_COL]
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # baseline: just predicting "same as last 3 avg" with no model at all
    baseline_mae = mean_absolute_error(y_test, test["avg_points_last_3"])

    print(f"Model MAE:    {mae:.3f} points")
    print(f"Baseline MAE: {baseline_mae:.3f} points (avg_points_last_3 with no model)")
    if mae < baseline_mae:
        print("Model beats the naive baseline.")
    else:
        print("Model does NOT beat the naive baseline yet - needs more features/data.")


def predict_next_gameweek(model, full_df, feature_cols, players_clean):
    """
    Build a features row for each player as of the LATEST gameweek played,
    to predict the NEXT gameweek's points.
    """
    latest = (
        full_df.sort_values("gameweek")
        .groupby("player_id")
        .tail(1)
        .copy()
    )

    # shift the rolling features forward one step: what happened in gw N
    # becomes "recent form" going into gw N+1
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
    latest["predicted_points"] = model.predict(X)

    result = latest[["player_id", "web_name", "position", "team_name", "predicted_points"]]
    result = result.merge(
        players_clean[["id", "price_m"]].rename(columns={"id": "player_id"}),
        on="player_id", how="left",
    )
    return result.sort_values("predicted_points", ascending=False)


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "training_table.csv")
    players_clean = pd.read_csv(DATA_DIR / "players_clean.csv")

    df_features, feature_cols = prepare_features(df)
    train, test = time_based_split(df_features, feature_cols)

    print(f"Train rows: {len(train)}, Test rows: {len(test)}")
    model = train_model(train, feature_cols)

    print("\n=== Evaluation ===")
    evaluate(model, test, feature_cols)

    joblib.dump(model, DATA_DIR / "points_model.pkl")
    print("\nModel saved to data/points_model.pkl")

    print("\n=== Predicting next gameweek ===")
    predictions = predict_next_gameweek(model, df_features, feature_cols, players_clean)
    predictions.to_csv(DATA_DIR / "predictions.csv", index=False)

    print("\nTop 15 predicted scorers:")
    print(predictions.head(15).to_string(index=False))
