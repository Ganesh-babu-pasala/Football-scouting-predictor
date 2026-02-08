import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

from pipeline import run_full_pipeline

# Page config (must be first Streamlit command)
st.set_page_config(layout="wide", page_title="Football Scouting Predictor")

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"

CLEANED_DATA_PATH = DATA_DIR / "CleanedDataset.csv"
YOUTH_TALENT_DATA_PATH = DATA_DIR / "YouthTalentProspects.csv"
MODEL_PATH = MODELS_DIR / "growth_model.joblib"
FEATURES_PATH = MODELS_DIR / "growth_features.joblib"

# ----------------------------
# Youth criteria (UI defaults)
# ----------------------------
DEFAULT_TARGET_AGE_MIN = 16
DEFAULT_TARGET_AGE_MAX = 19
DEFAULT_MIN_POTENTIAL = 75
DEFAULT_MIN_OVERALL = 60

# ----------------------------
# Session state init (no default prospect shown)
# ----------------------------
if "selected_prospect_row" not in st.session_state:
    st.session_state["selected_prospect_row"] = None

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model(model_path: Path):
    try:
        return joblib.load(model_path)
    except Exception:
        return None


@st.cache_resource
def load_features(features_path: Path):
    try:
        return joblib.load(features_path)
    except Exception:
        return None


# ----------------------------
# Helpers
# ----------------------------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Club" in df.columns:
        df["Club"] = df["Club"].fillna("Unknown Club").astype(str)
    return df


def ensure_growth_and_score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "Growth Potential" not in df.columns and {"Potential", "Overall"}.issubset(df.columns):
        df["Growth Potential"] = df["Potential"] - df["Overall"]

    if "ScoutScore" not in df.columns and {"Age", "Overall", "Growth Potential"}.issubset(df.columns):
        df["ScoutScore"] = (
            (df["Growth Potential"] * 0.6)
            + ((df["Overall"] - 50) * 0.3)
            + ((20 - df["Age"]) * 0.1)
        )

    return df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def predict_growth_for_row(model, features, row: pd.Series) -> float:
    if model is None or features is None:
        return np.nan
    try:
        X = pd.DataFrame([{f: row.get(f, 0) for f in features}]).replace([np.inf, -np.inf], np.nan).fillna(0)
        return float(model.predict(X)[0])
    except Exception:
        return np.nan


def get_player_row(df_main: pd.DataFrame, df_fallback: pd.DataFrame, player_name: str) -> pd.Series:
    if not df_main.empty and "Name" in df_main.columns:
        hit = df_main[df_main["Name"].astype(str) == str(player_name)]
        if not hit.empty:
            return hit.iloc[0]

    hit2 = df_fallback[df_fallback["Name"].astype(str) == str(player_name)]
    if not hit2.empty:
        return hit2.iloc[0]

    return df_fallback.iloc[0]


def top_skills_from_row(row: pd.Series, top_k: int = 8) -> pd.Series:
    skill_candidates = [
        "Acceleration", "Aggression", "Agility", "Balance", "Ball control", "Composure",
        "Crossing", "Curve", "Dribbling", "Finishing", "Free kick accuracy",
        "Heading accuracy", "Interceptions", "Jumping", "Long passing", "Long shots",
        "Marking", "Penalties", "Positioning", "Reactions", "Short passing",
        "Shot power", "Sliding tackle", "Sprint speed", "Stamina",
        "Standing tackle", "Strength", "Vision", "Volleys"
    ]
    skills = {}
    for s in skill_candidates:
        if s in row.index:
            v = row.get(s, np.nan)
            if pd.notna(v) and isinstance(v, (int, float, np.integer, np.floating)):
                skills[s] = float(v)

    if not skills:
        return pd.Series(dtype=float)

    return pd.Series(skills).nlargest(top_k)


def get_selected_row_from_event(event):
    try:
        sel = event.selection
        if sel and "rows" in sel and len(sel["rows"]) > 0:
            return int(sel["rows"][0])
    except Exception:
        pass
    return None


# ----------------------------
# Sidebar: navigation + pipeline controls
# ----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Player Explorer", "Youth Prospects", "Model Report"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Pipeline")

age_min = st.sidebar.number_input("Age min", min_value=14, max_value=40, value=DEFAULT_TARGET_AGE_MIN, step=1)
age_max = st.sidebar.number_input("Age max", min_value=14, max_value=40, value=DEFAULT_TARGET_AGE_MAX, step=1)
min_potential = st.sidebar.number_input("Min potential", min_value=0, max_value=100, value=DEFAULT_MIN_POTENTIAL, step=1)
min_overall = st.sidebar.number_input("Min overall", min_value=0, max_value=100, value=DEFAULT_MIN_OVERALL, step=1)
train_ml = st.sidebar.checkbox("Train ML model (predict Growth Potential)", value=True)

run_clicked = st.sidebar.button("Run pipeline and refresh")

if run_clicked:
    with st.spinner("Running pipeline..."):
        summary = run_full_pipeline(
            age_min=int(age_min),
            age_max=int(age_max),
            min_potential=int(min_potential),
            min_overall=int(min_overall),
            train_ml=bool(train_ml),
        )
    st.session_state["pipeline_summary"] = summary
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Pipeline completed.")

# ----------------------------
# Load data + model
# ----------------------------
df_main = ensure_columns(load_csv(CLEANED_DATA_PATH))
df_youth = ensure_columns(load_csv(YOUTH_TALENT_DATA_PATH))
df_youth = ensure_growth_and_score(df_youth)

model = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
features = load_features(FEATURES_PATH) if FEATURES_PATH.exists() else None

# ----------------------------
# App header
# ----------------------------
st.title("Football Scouting Predictor")

if "pipeline_summary" in st.session_state:
    s = st.session_state["pipeline_summary"]
    msg = f"Last pipeline run: cleaned {s['rows_cleaned']} rows, generated {s['youth_count']} youth prospects."
    if s.get("ml_trained") and s.get("ml_metrics"):
        m = s["ml_metrics"]
        msg += f" ML trained (MAE {m['mae']:.2f}, RMSE {m['rmse']:.2f})."
    st.caption(msg)

st.divider()

# ----------------------------
# Page: Dashboard
# ----------------------------
if page == "Dashboard":
    st.header("Dashboard")

    if df_main.empty:
        st.warning(
            "Cleaned dataset not found. Use the sidebar to run the pipeline. "
            "This will generate Data/CleanedDataset.csv and Data/YouthTalentProspects.csv."
        )
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Players", int(df_main.shape[0]))
    with col2:
        st.metric("Avg Overall", f"{df_main['Overall'].mean():.1f}" if "Overall" in df_main.columns else "N/A")
    with col3:
        st.metric("Avg Potential", f"{df_main['Potential'].mean():.1f}" if "Potential" in df_main.columns else "N/A")
    with col4:
        st.metric("Youth Prospects", int(df_youth.shape[0]) if not df_youth.empty else 0)

    left, right = st.columns(2)

    with left:
        if "Overall" in df_main.columns:
            fig = px.histogram(df_main, x="Overall", nbins=30, title="Overall distribution")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if {"Potential", "Overall"}.issubset(df_main.columns):
            tmp = df_main.copy()
            tmp["Growth Potential"] = tmp["Potential"] - tmp["Overall"]
            fig = px.histogram(tmp, x="Growth Potential", nbins=30, title="Growth Potential distribution")
            st.plotly_chart(fig, use_container_width=True)

    if {"Age", "Overall", "Potential"}.issubset(df_main.columns):
        tmp = df_main.copy()
        tmp["Growth Potential"] = tmp["Potential"] - tmp["Overall"]
        sample_n = min(3000, len(tmp))
        fig = px.scatter(
            tmp.sample(sample_n, random_state=42),
            x="Age",
            y="Growth Potential",
            hover_name="Name" if "Name" in tmp.columns else None,
            title="Age vs Growth Potential (sample)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Page: Player Explorer
# ----------------------------
elif page == "Player Explorer":
    st.header("Player Explorer")

    if df_main.empty:
        st.error("Cleaned dataset not found. Run the pipeline first.")
        st.stop()

    search = st.text_input("Search by player name", value="")
    filtered = df_main
    if search.strip():
        filtered = df_main[df_main["Name"].astype(str).str.contains(search, case=False, na=False)]

    if filtered.empty:
        st.warning("No matching players found.")
        st.stop()

    selected = st.selectbox("Select player", filtered["Name"].sort_values().tolist())
    row = filtered[filtered["Name"] == selected].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall", int(row.get("Overall", 0)))
        st.metric("Potential", int(row.get("Potential", 0)))
    with c2:
        st.write(f"Club: {row.get('Club', 'Unknown')}")
        st.write(f"Nationality: {row.get('Nationality', 'Unknown')}")
        st.write(f"Age: {int(row.get('Age', 0))}")
    with c3:
        val = row.get("Value", np.nan)
        wage = row.get("Wage", np.nan)
        st.metric("Estimated Value", f"{val:,.0f}" if pd.notna(val) else "N/A")
        st.metric("Weekly Wage", f"{wage:,.0f}" if pd.notna(wage) else "N/A")

    st.divider()

    pred_growth = predict_growth_for_row(model, features, row) if model is not None else np.nan
    if pd.notna(pred_growth):
        st.subheader("ML Prediction")
        st.write(f"Predicted Growth Potential: {pred_growth:.2f}")
        if pd.notna(row.get("Overall", np.nan)):
            st.write(f"Estimated Future Potential (Overall + Predicted Growth): {float(row['Overall']) + pred_growth:.2f}")
    else:
        st.info("ML model not available. Run the pipeline with ML training enabled to generate predictions.")

    st.subheader("Top skills")
    top = top_skills_from_row(row, top_k=8)
    if not top.empty:
        fig = px.bar(x=top.index, y=top.values, labels={"x": "Skill", "y": "Rating"}, title="Top skills")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No skill columns available for this player.")

# ----------------------------
# Page: Youth Prospects (no default details shown)
# ----------------------------
elif page == "Youth Prospects":
    st.header("Youth Prospects")

    if df_youth.empty:
        st.warning("Youth prospects file not found. Run the pipeline first.")
        st.stop()

    left, right = st.columns([1, 3])

    with left:
        st.subheader("Filters")
        pot_min = int(df_youth["Potential"].min()) if "Potential" in df_youth.columns else 0
        pot_max = int(df_youth["Potential"].max()) if "Potential" in df_youth.columns else 100
        gp_min = int(df_youth["Growth Potential"].min()) if "Growth Potential" in df_youth.columns else 0
        gp_max = int(df_youth["Growth Potential"].max()) if "Growth Potential" in df_youth.columns else 50

        f_min_pot = st.slider("Minimum Potential", min_value=pot_min, max_value=pot_max, value=pot_min)
        f_min_gp = st.slider("Minimum Growth Potential", min_value=gp_min, max_value=gp_max, value=gp_min)
        sort_by = st.selectbox("Sort by", ["ScoutScore", "Growth Potential", "Potential", "Overall"], index=0)
        top_n = st.selectbox("Show top N", [25, 50, 100, 200], index=0)

    filtered = df_youth.copy()
    if "Potential" in filtered.columns:
        filtered = filtered[filtered["Potential"] >= f_min_pot]
    if "Growth Potential" in filtered.columns:
        filtered = filtered[filtered["Growth Potential"] >= f_min_gp]
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=False)

    filtered = filtered.reset_index(drop=True)

    display_cols = [
        "Name", "Age", "Nationality", "Club",
        "Overall", "Potential", "Growth Potential", "ScoutScore",
        "Value", "Wage", "Preferred Positions"
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    with right:
        st.subheader(f"Results ({filtered.shape[0]} prospects)")

        selected_index = None
        try:
            event = st.dataframe(
                filtered[display_cols].head(top_n),
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=False
            )
            selected_index = get_selected_row_from_event(event)
        except TypeError:
            st.dataframe(filtered[display_cols].head(top_n), use_container_width=True)

        st.download_button(
            "Download filtered prospects (CSV)",
            data=df_to_csv_bytes(filtered[display_cols]),
            file_name="youth_prospects_filtered.csv",
            mime="text/csv",
        )



# ----------------------------
# Page: Model Report
# ----------------------------
else:
    st.header("Model Report")

    if model is None or features is None:
        st.info("ML model not found. Run the pipeline with ML training enabled to generate the model.")
        st.stop()

    st.subheader("Feature importance")
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            st.write("This model does not expose feature importances.")
        else:
            imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
            imp_df = imp_df.sort_values(by="Importance", ascending=False).head(20)
            fig = px.bar(imp_df[::-1], x="Importance", y="Feature", orientation="h", title="Top 20 features")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render feature importance: {e}")

    if "pipeline_summary" in st.session_state and st.session_state["pipeline_summary"].get("ml_metrics"):
        m = st.session_state["pipeline_summary"]["ml_metrics"]
        st.subheader("Latest training metrics")
        st.write(f"MAE: {m['mae']:.2f}")
        st.write(f"RMSE: {m['rmse']:.2f}")
        st.write(f"Rows used: {m['n_rows']}")
        st.write(f"Number of features: {m['n_features']}")
