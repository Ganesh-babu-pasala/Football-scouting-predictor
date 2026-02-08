from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

# ML (optional but included)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"

RAW_PATH = DATA_DIR / "CompleteDataset.csv"
CLEANED_PATH = DATA_DIR / "CleanedDataset.csv"
YOUTH_PATH = DATA_DIR / "YouthTalentProspects.csv"

MODEL_PATH = MODELS_DIR / "growth_model.joblib"
FEATURES_PATH = MODELS_DIR / "growth_features.joblib"
METRICS_PATH = MODELS_DIR / "growth_metrics.json"


# ----------------------------
# Helpers
# ----------------------------
def clean_money_value(value) -> float:
    """Convert strings like '£28.5M'/'£975K' to numeric."""
    if pd.isna(value):
        return 0.0
    s = (
        str(value)
        .replace("£", "")
        .replace("€", "")
        .replace("$", "")
        .replace(",", "")
        .replace("�", "")
        .strip()
    )
    try:
        if "M" in s:
            return float(s.replace("M", "")) * 1_000_000
        if "K" in s:
            return float(s.replace("K", "")) * 1_000
        return float(s)
    except ValueError:
        return 0.0


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


# ----------------------------
# Step 1: Clean dataset
# ----------------------------
def clean_dataset() -> pd.DataFrame:
    """
    Load raw FIFA dataset and produce CleanedDataset.csv.
    Keeps things simple and robust for demo use.
    """
    ensure_dirs()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw dataset: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, low_memory=False, encoding_errors="ignore")

    redundant_cols = ["Unnamed: 0", "Photo", "Flag", "Club Logo", "ID"]
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns], errors="ignore")

    if "Value" in df.columns:
        df["Value"] = df["Value"].apply(clean_money_value)
    if "Wage" in df.columns:
        df["Wage"] = df["Wage"].apply(clean_money_value)

    positional_cols = [
        "CAM", "CB", "CDM", "CF", "CM", "LAM", "LB", "LCB", "LCM", "LDM",
        "LF", "LM", "LS", "LW", "LWB", "RAM", "RB", "RCB", "RCM", "RDM",
        "RF", "RM", "RS", "RW", "RWB", "ST"
    ]
    for col in positional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Convert common skill columns to numeric (if present)
    skill_cols = [
        "Acceleration", "Aggression", "Agility", "Balance", "Ball control", "Composure",
        "Crossing", "Curve", "Dribbling", "Finishing", "Free kick accuracy",
        "Heading accuracy", "Interceptions", "Jumping", "Long passing", "Long shots",
        "Marking", "Penalties", "Positioning", "Reactions", "Short passing",
        "Shot power", "Sliding tackle", "Sprint speed", "Stamina",
        "Standing tackle", "Strength", "Vision", "Volleys"
    ]
    for col in skill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median()).astype(float)

    # Fill remaining numeric NaNs except key columns
    exclude_cols = ["Age", "Overall", "Potential", "Value", "Wage"] + positional_cols
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col not in exclude_cols and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df.to_csv(CLEANED_PATH, index=False)
    return df


# ----------------------------
# Step 2: Youth prospects
# ----------------------------
def build_youth_prospects(
    df: pd.DataFrame,
    age_min: int = 16,
    age_max: int = 19,
    min_potential: int = 75,
    min_overall: int = 60,
) -> pd.DataFrame:
    """Create YouthTalentProspects.csv including Growth Potential and ScoutScore."""
    ensure_dirs()

    youth = df[
        (df["Age"] >= age_min) &
        (df["Age"] <= age_max) &
        (df["Potential"] >= min_potential) &
        (df["Overall"] >= min_overall)
    ].copy()

    youth["Growth Potential"] = youth["Potential"] - youth["Overall"]
    youth["ScoutScore"] = (
        (youth["Growth Potential"] * 0.6) +
        ((youth["Overall"] - 50) * 0.3) +
        ((20 - youth["Age"]) * 0.1)
    )

    youth = youth.sort_values(by=["ScoutScore", "Growth Potential", "Potential"], ascending=False)
    youth.to_csv(YOUTH_PATH, index=False)
    return youth


# ----------------------------
# Step 3: Train ML model for Growth Potential
# ----------------------------
def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Choose numeric feature columns for predicting Growth Potential.
    Excludes target leakage columns.
    """
    leak_cols = {
        "Potential", "Growth Potential", "GrowthPotential", "ScoutScore"
    }
    # Use numeric columns except obvious IDs
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in leak_cols]

    # Keep Age and Overall if available (they are useful)
    # They are numeric, so will already be included.
    return feature_cols


def train_growth_model(df: pd.DataFrame, random_state: int = 42) -> Tuple[RandomForestRegressor, dict]:
    """
    Train a regression model to predict Growth Potential from player attributes.
    Saves model + features to models/.
    """
    ensure_dirs()

    df = df.copy()
    df["Growth Potential"] = df["Potential"] - df["Overall"]

    feature_cols = choose_feature_columns(df)
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Growth Potential"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metrics = {"mae": mae, "rmse": rmse, "n_features": len(feature_cols), "n_rows": int(df.shape[0])}

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    return model, metrics


# ----------------------------
# Full pipeline
# ----------------------------
def run_full_pipeline(
    age_min: int = 16,
    age_max: int = 19,
    min_potential: int = 75,
    min_overall: int = 60,
    train_ml: bool = True,
) -> dict:
    """
    Run cleaning -> youth prospects -> optional ML training.
    Returns summary dict for displaying in Streamlit.
    """
    df = clean_dataset()
    youth = build_youth_prospects(df, age_min, age_max, min_potential, min_overall)

    out = {
        "rows_cleaned": int(df.shape[0]),
        "cols_cleaned": int(df.shape[1]),
        "youth_count": int(youth.shape[0]),
        "cleaned_path": str(CLEANED_PATH),
        "youth_path": str(YOUTH_PATH),
        "ml_trained": False,
        "ml_metrics": None,
    }

    if train_ml:
        _, metrics = train_growth_model(df)
        out["ml_trained"] = True
        out["ml_metrics"] = metrics

    return out
