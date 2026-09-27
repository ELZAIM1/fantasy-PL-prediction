"""
Streamlit dashboard for the FPL predictor.

Run with: streamlit run app.py
(from the same folder as your data/ directory and the other scripts)
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from optimize_squad import pick_squad, pick_best_xi, BUDGET

DATA_DIR = Path("data")

st.set_page_config(page_title="FPL Squad Predictor", layout="wide")


@st.cache_data
def load_predictions():
    df = pd.read_csv(DATA_DIR / "predictions.csv")
    df["position"] = df["position"].replace({"GK": "GKP"})
    return df


def render_pitch(xi, captain):
    """Simple text-based pitch layout, grouped by position."""
    for pos, label in [("GKP", "Goalkeeper"), ("DEF", "Defenders"),
                        ("MID", "Midfielders"), ("FWD", "Forwards")]:
        row = xi[xi["position"] == pos]
        if row.empty:
            continue
        cols = st.columns(len(row))
        for col, (_, player) in zip(cols, row.iterrows()):
            with col:
                is_cap = player["player_id"] == captain["player_id"]
                name = f"**{player['web_name']}** {'(C)' if is_cap else ''}"
                pts = player["predicted_points"] * (2 if is_cap else 1)
                st.markdown(name)
                st.caption(f"{player['team_name']} | {pts:.1f} pts")


st.title("⚽ FPL Squad Predictor")

try:
    predictions = load_predictions()
except FileNotFoundError:
    st.error(
        "No predictions.csv found in data/. Run fetch_data.py, load_data.py, "
        "build_training_data.py, and train_model.py first."
    )
    st.stop()

tab1, tab2 = st.tabs(["📊 Player Predictions", "🏆 Optimal Squad"])

# ---------------- Tab 1: browse predictions ----------------
with tab1:
    st.subheader("Predicted points for the next gameweek")

    col1, col2, col3 = st.columns(3)
    with col1:
        pos_filter = st.multiselect(
            "Position", options=predictions["position"].unique(),
            default=list(predictions["position"].unique()),
        )
    with col2:
        team_filter = st.multiselect(
            "Team", options=sorted(predictions["team_name"].unique()),
        )
    with col3:
        max_price = st.slider(
            "Max price (£m)", min_value=float(predictions["price_m"].min()),
            max_value=float(predictions["price_m"].max()),
            value=float(predictions["price_m"].max()),
        )

    filtered = predictions[
        predictions["position"].isin(pos_filter) & (predictions["price_m"] <= max_price)
    ]
    if team_filter:
        filtered = filtered[filtered["team_name"].isin(team_filter)]

    st.dataframe(
        filtered.sort_values("predicted_points", ascending=False)
        [["web_name", "position", "team_name", "price_m", "predicted_points"]],
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        filtered.nlargest(15, "predicted_points").set_index("web_name")["predicted_points"]
    )

# ---------------- Tab 2: run the optimizer ----------------
with tab2:
    st.subheader("Best possible squad under FPL rules")
    st.caption(f"Budget £{BUDGET}m · 2 GKP / 5 DEF / 5 MID / 3 FWD · max 3 per club")

    if st.button("Optimize squad", type="primary"):
        with st.spinner("Solving..."):
            squad = pick_squad(predictions)
            xi, formation, captain, bench = pick_best_xi(squad)

        m1, m2, m3 = st.columns(3)
        m1.metric("Squad cost", f"£{squad['price_m'].sum():.1f}m / £{BUDGET}m")
        m2.metric("Formation", formation)
        m3.metric("Predicted XI points (with captain)",
                   f"{xi['predicted_points'].sum() + captain['predicted_points']:.1f}")

        st.markdown("### Starting XI")
        render_pitch(xi, captain)

        st.markdown("### Bench")
        st.dataframe(
            bench[["web_name", "position", "team_name", "price_m", "predicted_points"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Click the button to run the optimizer.")
