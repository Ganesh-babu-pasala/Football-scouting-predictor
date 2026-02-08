import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
CLEANED_DATA_PATH = "Data/CleanedDataset.csv"
OUTPUT_YOUTH_TALENT_PATH = "Data/YouthTalentProspects.csv"

# Youth scouting criteria
TARGET_AGE_MIN = 16
TARGET_AGE_MAX = 19
MIN_POTENTIAL_RATING = 75
MIN_OVERALL_RATING = 60


# --------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------
def load_cleaned_data(path: str) -> pd.DataFrame:
    """Load the cleaned FIFA dataset."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded cleaned dataset ({df.shape[0]} rows).")
        return df
    except FileNotFoundError:
        print(f"File not found: {path}. Run 01_data_exploration.py first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()


df = load_cleaned_data(CLEANED_DATA_PATH)

if df.empty:
    raise SystemExit("Stopping: cleaned dataset could not be loaded.")


# --------------------------------------------------
# Identify youth talent prospects
# --------------------------------------------------
print("Identifying youth talent prospects...")

youth_df = df[
    (df["Age"] >= TARGET_AGE_MIN) &
    (df["Age"] <= TARGET_AGE_MAX) &
    (df["Potential"] >= MIN_POTENTIAL_RATING) &
    (df["Overall"] >= MIN_OVERALL_RATING)
].copy()

# Growth Potential metric
youth_df["Growth Potential"] = youth_df["Potential"] - youth_df["Overall"]

# Simple, interpretable scouting score
youth_df["ScoutScore"] = (
    (youth_df["Growth Potential"] * 0.6) +
    ((youth_df["Overall"] - 50) * 0.3) +
    ((20 - youth_df["Age"]) * 0.1)
)

# Rank prospects
youth_df = youth_df.sort_values(
    by=["ScoutScore", "Growth Potential", "Potential"],
    ascending=False
)

print(f"Total youth prospects identified: {youth_df.shape[0]}")


# --------------------------------------------------
# Preview top prospects
# --------------------------------------------------
display_cols = [
    "Name", "Age", "Nationality", "Club",
    "Overall", "Potential", "Growth Potential",
    "ScoutScore", "Value", "Wage", "Preferred Positions"
]
display_cols = [c for c in display_cols if c in youth_df.columns]

print("\nTop 10 youth prospects:")
print(youth_df[display_cols].head(10).to_string(index=False))


# --------------------------------------------------
# Save output
# --------------------------------------------------
if not youth_df.empty:
    youth_df.to_csv(OUTPUT_YOUTH_TALENT_PATH, index=False)
    print(f"Saved youth prospects to {OUTPUT_YOUTH_TALENT_PATH}")
else:
    print("No youth prospects found. Output file not created.")

print("Youth talent identification completed.")
