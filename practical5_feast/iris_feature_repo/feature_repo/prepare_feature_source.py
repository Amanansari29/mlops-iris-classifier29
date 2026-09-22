"""Prepare the Experiment 4 CSV as a Feast-ready Parquet source."""

from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = REPOSITORY_ROOT / "data" / "processed" / "iris_features.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "iris_features.parquet"


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df.insert(0, "sample_id", range(len(df)))

    start_time = pd.Timestamp("2026-08-15 15:20:02", tz="UTC")
    df["event_timestamp"] = pd.date_range(
        start=start_time, periods=len(df), freq="min"
    )
    df["created_timestamp"] = df["event_timestamp"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
