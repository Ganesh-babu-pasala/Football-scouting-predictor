import pandas as pd
import numpy as np
import chardet

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_PATH = "Data/CompleteDataset.csv"
OUTPUT_CLEANED_PATH = "Data/CleanedDataset.csv"


# --------------------------------------------------
# Detect file encoding
# --------------------------------------------------
def detect_encoding(path: str) -> str:
    """Detect file encoding using chardet."""
    with open(path, "rb") as f:
        result = chardet.detect(f.read(100000))

    encoding = result.get("encoding")
    confidence = result.get("confidence", 0)

    if encoding is None or confidence < 0.8:
        print("Low confidence encoding detection. Falling back to utf-8.")
        return "utf-8"

    print(f"Detected encoding: {encoding} (confidence {confidence:.2f})")
    return encoding


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("Loading dataset...")

try:
    encoding = detect_encoding(DATA_PATH)
    df = pd.read_csv(
        DATA_PATH,
        encoding=encoding,
        encoding_errors="ignore",
        low_memory=False
    )
except Exception as e:
    raise SystemExit(f"Failed to load dataset: {e}")

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def clean_money_value(value):
    """
    Convert currency strings like '£28.5M' or '£975K' to numeric values.
    """
    if pd.isna(value):
        return 0.0

    value = (
        str(value)
        .replace("£", "")
        .replace("€", "")
        .replace("$", "")
        .replace(",", "")
        .replace("�", "")
        .strip()
    )

    try:
        if "M" in value:
            return float(value.replace("M", "")) * 1_000_000
        if "K" in value:
            return float(value.replace("K", "")) * 1_000
        return float(value)
    except ValueError:
        return 0.0


# --------------------------------------------------
# Data cleaning
# --------------------------------------------------
print("Cleaning data...")

# Drop redundant columns
redundant_cols = ["Unnamed: 0", "Photo", "Flag", "Club Logo", "ID"]
df = df.drop(columns=[c for c in redundant_cols if c in df.columns], errors="ignore")

# Clean Value and Wage
if "Value" in df.columns:
    df["Value"] = df["Value"].apply(clean_money_value)

if "Wage" in df.columns:
    df["Wage"] = df["Wage"].apply(clean_money_value)

# Positional rating columns
positional_cols = [
    "CAM", "CB", "CDM", "CF", "CM", "LAM", "LB", "LCB", "LCM", "LDM",
    "LF", "LM", "LS", "LW", "LWB", "RAM", "RB", "RCB", "RCM", "RDM",
    "RF", "RM", "RS", "RW", "RWB", "ST"
]

for col in positional_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(float)

# Convert skill attributes to numeric
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

# Fill remaining numeric NaNs (excluding key ratings)
exclude_cols = ["Age", "Overall", "Potential", "Value", "Wage"] + positional_cols
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    if col not in exclude_cols and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

print("Data cleaning complete.")


# --------------------------------------------------
# Basic checks
# --------------------------------------------------
print("Post-cleaning summary:")
print(df[["Age", "Overall", "Potential"]].describe())

print("Top rows preview:")
print(df[["Name", "Age", "Overall", "Potential", "Club", "Value", "Wage"]].head())


# --------------------------------------------------
# Save cleaned dataset
# --------------------------------------------------
df.to_csv(OUTPUT_CLEANED_PATH, index=False)
print(f"Cleaned dataset saved to {OUTPUT_CLEANED_PATH}")
