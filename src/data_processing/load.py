"""
Data loading utilities for raw data from the data/raw/ folder.
Supports loading train, test, and holdout datasets.
"""

from pathlib import Path

import pandas as pd


def load_data(file_path) -> pd.DataFrame:
    """
    Load training dataset from data/raw/train.csv

    Returns:
        pd.DataFrame: Training data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Train data not found: {file_path}")

    print(f"Loading training data from {file_path}...")

    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} training samples with {len(df.columns)} features")

    return df
