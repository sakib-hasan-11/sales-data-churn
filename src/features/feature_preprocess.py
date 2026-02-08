from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_features(
    df: pd.DataFrame, output_path: str | Path = None, name: str = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Preprocess feature data:
    - Encode categorical variables
    - One-hot encode grouped categories
    - Scale numerical features
    - Optionally save processed dataset

    Returns:
        tuple: (X, y, feature_names) where:
            - X: numpy array of features
            - y: numpy array of target variable (churn)
            - feature_names: list of feature column names
    """

    df = df.copy()

    # =====================================================
    # 1️ LABEL ENCODE GENDER (already normalized)
    # =====================================================
    le_gender = LabelEncoder()
    df["gender"] = le_gender.fit_transform(df["gender"])

    # =====================================================
    # 2️ ONE-HOT ENCODING (normalized column names)
    # =====================================================

    subscription_dummies = pd.get_dummies(df["subscription_type"], prefix="sub")
    contract_dummies = pd.get_dummies(df["contract_length"], prefix="contract")
    tenure_dummies = pd.get_dummies(df["tenure_category"], prefix="tenuregroup")
    age_dummies = pd.get_dummies(df["age_group"], prefix="agegroup")
    spend_dummies = pd.get_dummies(df["spend_category"], prefix="spendcategory")

    df = pd.concat(
        [
            df,
            subscription_dummies,
            contract_dummies,
            tenure_dummies,
            age_dummies,
            spend_dummies,
        ],
        axis=1,
    )

    # =====================================================
    # 3️ DROP ORIGINAL CATEGORICAL COLUMNS
    # =====================================================
    cols_to_drop = [
        "subscription_type",
        "contract_length",
        "tenure_category",
        "age_group",
        "spend_category",
    ]

    # Only drop customerid if it exists
    if "customerid" in df.columns:
        cols_to_drop.append("customerid")

    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # =====================================================
    # 4️ SEPARATE TARGET VARIABLE
    # =====================================================
    y = df["churn"].values
    df = df.drop(columns=["churn"])

    # =====================================================
    # 5️ SCALE NUMERICAL FEATURES
    # =====================================================
    excluded_cols = {"gender"}  # gender already encoded, churn already removed
    features_to_scale = [
        col
        for col in df.columns
        if col not in excluded_cols and np.issubdtype(df[col].dtype, np.number)
    ]

    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    print(f"Scaled {len(features_to_scale)} numerical features")

    # =====================================================
    # 6️ PREPARE OUTPUT
    # =====================================================
    X = df.values
    feature_names = df.columns.tolist()

    # =====================================================
    # 7️ OPTIONALLY SAVE OUTPUT
    # =====================================================
    if output_path is not None and name is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save with churn column for completeness
        df_to_save = df.copy()
        df_to_save["churn"] = y
        output_file = output_path / name
        df_to_save.to_csv(output_file, index=False)

    return X, y, feature_names
