"""
Prepare dummy data for CI testing
Split dummy_data.csv into train and test sets for pipeline testing
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def prepare_ci_data():
    """Split dummy data into train and test for CI pipeline testing"""

    data_dir = project_root / "data" / "raw"
    dummy_file = data_dir / "dummy_data.csv"

    print(f"Loading dummy data from: {dummy_file}")

    # Load dummy data
    df = pd.read_csv(dummy_file)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Drop rows with missing target variable (Churn)
    if "Churn" in df.columns:
        initial_count = len(df)
        df = df.dropna(subset=["Churn"])
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing Churn values")
            print(f"Remaining samples: {len(df)}")

    # Split into train and test (70/30)
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["Churn"] if "Churn" in df.columns else None,
    )

    # Save to data/raw/
    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\nData split complete:")
    print(f"  Training samples: {len(train_df)} -> {train_file}")
    print(f"  Test samples: {len(test_df)} -> {test_file}")
    print(f"\nTrain distribution:")
    if "Churn" in train_df.columns:
        print(train_df["Churn"].value_counts())
    print(f"\nTest distribution:")
    if "Churn" in test_df.columns:
        print(test_df["Churn"].value_counts())

    return train_file, test_file


if __name__ == "__main__":
    prepare_ci_data()
