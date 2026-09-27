"""
Step 4: Train a SEPARATE model for each of the next HORIZON gameweeks

(GW+1, GW+2, ... GW+HORIZON), all using the same "current form" features.

This gives a week-by-week predicted breakdown, not just one lump total -
useful for spotting who's peaking soon vs. who's got a tough patch coming,
not just "who's good overall."

The squad optimizer still gets one number per player:
the sum across all available HORIZON models' predictions.

Early in the season, some future horizons may not have enough completed
gameweeks to train a model. Those horizons are skipped instead of crashing
the whole pipeline.
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

TARGET_COLS = [
    f"target_gw_plus_{i}"
    for i in range(1, HORIZON + 1)
]


def prepare_features(df):
    """
    Prepare model features.

    Keeps the original 'position' column for readable output and
    one-hot encodes it for the model.
    """

    df = df.copy()

    df["was_home"] = df["was_home"].astype(int)

    # One-hot encode position.
    # Keep the original position column too.
    pos_dummies = pd.get_dummies(
        df["position"],
        prefix="pos"
    )

    df = pd.concat(
        [df, pos_dummies],
        axis=1
    )

    pos_dummy_cols = list(pos_dummies.columns)

    feature_cols = FEATURE_COLS + pos_dummy_cols

    return df, feature_cols


def time_based_split(
    df,
    feature_cols,
    target_col,
    test_gws=3
):
    """
    Don't shuffle-split time series data.

    Train on earlier gameweeks and test on the most recent gameweeks.

    Early in the season there may only be a small number of complete
    gameweeks for a particular target. In that case, raise ValueError
    so the caller can skip that horizon.
    """

    # Keep only rows where both features and target are available.
    df = df.dropna(
        subset=feature_cols + [target_col]
    )

    unique_gws = sorted(
        df["gameweek"].unique()
    )

    n_gws = len(unique_gws)

    if n_gws < 2:
        raise ValueError(
            f"Only {n_gws} gameweek(s) have complete data "
            f"for {target_col} (need at least 2)."
        )

    # Never use all gameweeks for training.
    # Keep at least one GW for testing.
    test_gws = min(
        test_gws,
        n_gws - 1
    )

    cutoff = unique_gws[-(test_gws + 1)]

    train = df[
        df["gameweek"] <= cutoff
    ]

    test = df[
        df["gameweek"] > cutoff
    ]

    return train, test


def train_one_model(
    train,
    feature_cols,
    target_col
):
    """
    Train one Gradient Boosting model for one horizon.
    """

    X_train = train[feature_cols]
    y_train = train[target_col]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_one(
    model,
    test,
    feature_cols,
    target_col,
    label
):
    """
    Evaluate the model against a simple recent-form baseline.
    """

    X_test = test[feature_cols]
    y_test = test[target_col]

    preds = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    # Naive baseline:
    # assume recent average form repeats.
    baseline_mae = mean_absolute_error(
        y_test,
        test["avg_points_last_3"]
    )

    beats = (
        "beats"
        if mae < baseline_mae
        else "does NOT beat"
    )

    print(
        f"  {label}: "
        f"model MAE={mae:.3f}  "
        f"baseline MAE={baseline_mae:.3f}  "
        f"({beats} baseline)"
    )

    return mae, baseline_mae


def predict_upcoming_horizon(
    models,
    full_df,
    feature_cols,
    players_clean
):
    """
    Build one feature row per player using the latest available data,
    then run every available horizon model.

    models is a dictionary:

        {
            1: model_for_GW_plus_1,
            2: model_for_GW_plus_2,
            ...
        }

    This preserves the correct GW offset even when some horizons
    are unavailable early in the season.
    """

    # ---------------------------------------------------------
    # Get latest row for every player
    # ---------------------------------------------------------

    latest = (
        full_df
        .sort_values("gameweek")
        .groupby("player_id")
        .tail(1)
        .copy()
    )

    # ---------------------------------------------------------
    # Recalculate current-form features
    # ---------------------------------------------------------

    sorted_df = full_df.sort_values(
        ["player_id", "gameweek"]
    ).copy()

    points_rolling = (
        sorted_df
        .groupby("player_id")["total_points"]
        .rolling(3)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    minutes_rolling = (
        sorted_df
        .groupby("player_id")["minutes"]
        .rolling(3)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    sorted_df["avg_points_last_3"] = points_rolling
    sorted_df["avg_minutes_last_3"] = minutes_rolling

    # Map latest rolling values back to latest player rows.
    latest_form = (
        sorted_df
        .groupby("player_id")
        .tail(1)
        .set_index("player_id")
    )

    latest["avg_points_last_3"] = (
        latest["player_id"]
        .map(latest_form["avg_points_last_3"])
    )

    latest["avg_minutes_last_3"] = (
        latest["player_id"]
        .map(latest_form["avg_minutes_last_3"])
    )

    latest["points_last_gw"] = latest["total_points"]

    # ---------------------------------------------------------
    # Current code does not map the upcoming fixture yet.
    # Keep the original neutral/default behavior.
    # ---------------------------------------------------------

    latest["was_home"] = 0

    # ---------------------------------------------------------
    # Drop rows with missing required features
    # ---------------------------------------------------------

    latest = latest.dropna(
        subset=feature_cols
    )

    X = latest[feature_cols]

    # ---------------------------------------------------------
    # Predict every available horizon
    # ---------------------------------------------------------

    gw_pred_cols = []

    for i, model in sorted(models.items()):

        col_name = f"pred_gw_plus_{i}"

        latest[col_name] = model.predict(X)

        gw_pred_cols.append(
            col_name
        )

    # ---------------------------------------------------------
    # Sum predictions across available horizons
    # ---------------------------------------------------------

    if gw_pred_cols:
        latest["predicted_points"] = (
            latest[gw_pred_cols]
            .sum(axis=1)
        )
    else:
        latest["predicted_points"] = 0.0

    # ---------------------------------------------------------
    # Keep readable output columns
    # ---------------------------------------------------------

    keep_cols = [
        "player_id",
        "web_name",
        "position",
        "team_name",
    ]

    keep_cols += gw_pred_cols

    keep_cols.append(
        "predicted_points"
    )

    result = latest[keep_cols]

    # ---------------------------------------------------------
    # Add player price
    # ---------------------------------------------------------

    result = result.merge(
        players_clean[
            ["id", "price_m"]
        ].rename(
            columns={
                "id": "player_id"
            }
        ),
        on="player_id",
        how="left",
    )

    return result.sort_values(
        "predicted_points",
        ascending=False
    )


if __name__ == "__main__":

    # =========================================================
    # 1. Load data
    # =========================================================

    df = pd.read_csv(
        DATA_DIR / "training_table.csv"
    )

    players_clean = pd.read_csv(
        DATA_DIR / "players_clean.csv"
    )

    # =========================================================
    # 2. Prepare features
    # =========================================================

    df_features, feature_cols = prepare_features(
        df
    )

    # =========================================================
    # 3. Train one model per available horizon
    # =========================================================

    models = {}

    print(
        f"Training up to {HORIZON} models "
        f"(one per gameweek offset)..."
    )

    for i, target_col in enumerate(
        TARGET_COLS,
        start=1
    ):

        print(
            f"\n--- GW+{i} ---"
        )

        try:

            train, test = time_based_split(
                df_features,
                feature_cols,
                target_col
            )

        except ValueError as e:

            print(
                f"Skipping GW+{i}: {e}"
            )

            continue

        # Train
        model = train_one_model(
            train,
            feature_cols,
            target_col
        )

        # Store using the actual horizon number.
        models[i] = model

        # Evaluate
        evaluate_one(
            model,
            test,
            feature_cols,
            target_col,
            label=f"GW+{i}"
        )

        print(
            f"  Train rows: {len(train)}"
        )

        print(
            f"  Test rows:  {len(test)}"
        )

    # =========================================================
    # 4. Make sure at least one model exists
    # =========================================================

    if not models:

        raise RuntimeError(
            "No models could be trained. "
            "Not enough completed gameweeks yet."
        )

    # =========================================================
    # 5. Save models
    # =========================================================

    joblib.dump(
        models,
        DATA_DIR / "points_models.pkl"
    )

    print(
        "\nModels saved to "
        "data/points_models.pkl"
    )

    print(
        "\nAvailable models:"
    )

    for i in sorted(models):
        print(
            f"  GW+{i}"
        )

    # =========================================================
    # 6. Predict upcoming horizon
    # =========================================================

    print(
        f"\n=== Predicting available "
        f"gameweeks ==="
    )

    predictions = predict_upcoming_horizon(
        models,
        df_features,
        feature_cols,
        players_clean
    )

    # =========================================================
    # 7. Save predictions
    # =========================================================

    predictions.to_csv(
        DATA_DIR / "predictions.csv",
        index=False
    )

    print(
        "\nPredictions saved to "
        "data/predictions.csv"
    )

    # =========================================================
    # 8. Display top predicted players
    # =========================================================

    print(
        f"\nTop 15 predicted scorers "
        f"(total over available horizons):"
    )

    print(
        predictions
        .head(15)
        .to_string(index=False)
    )