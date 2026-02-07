import pandas as pd
import os

RAW_DATA_PATH = "data/raw/activity_log_raw.csv"
PROCESSED_DATA_PATH = "data/processed/activity_features.csv"


def label_activity(row):
    """
    Rule-based labeling logic
    """

    # IDLE
    if row["idle_time"] >= 30 and row["activity_intensity"] < 10:
        return "Idle"

    # PRODUCTIVE
    if (
        row["key_presses"] >= 10
        and row["switch_rate"] <= 1
        and row["idle_time"] < 10
    ):
        return "Productive"

    # FAKE PRODUCTIVE
    return "Fake_Productive"


def main():
    print("Starting feature engineering...")

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # -------- FEATURE ENGINEERING -------- #

    # Total interaction
    df["activity_intensity"] = (
        df["mouse_distance"] * 0.01 + df["key_presses"]
    )

    # Avoid division by zero
    df["typing_ratio"] = df.apply(
        lambda x: x["key_presses"] / x["activity_intensity"]
        if x["activity_intensity"] > 0 else 0,
        axis=1
    )

    # Window switch rate (per interval)
    df["switch_rate"] = df["window_switches"]

    # -------- LABELING -------- #
    df["label"] = df.apply(label_activity, axis=1)

    # -------- SELECT FINAL FEATURES -------- #
    final_df = df[
        [
            "mouse_distance",
            "mouse_clicks",
            "key_presses",
            "idle_time",
            "activity_intensity",
            "typing_ratio",
            "switch_rate",
            "label",
        ]
    ]

    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Feature engineering complete.")
    print("Saved to:", PROCESSED_DATA_PATH)
    print("\nLabel distribution:")
    print(final_df["label"].value_counts())


if __name__ == "__main__":
    main()
