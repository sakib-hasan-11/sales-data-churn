from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_features(
    df: pd.DataFrame,
    output_path: str | Path,
    name: str
) -> pd.DataFrame:
    """
    Preprocess feature data:
    - Encode categorical variables
    - One-hot encode grouped categories
    - Scale numerical features
    - Save processed dataset
    """

    df = df.copy()

    # =====================================================
    # 1️⃣ LABEL ENCODE GENDER (already normalized)
    # =====================================================
    le_gender = LabelEncoder()
    df["gender"] = le_gender.fit_transform(df["gender"])

    # =====================================================
    # 2️⃣ ONE-HOT ENCODING (normalized column names)
    # =====================================================

    subscription_dummies = pd.get_dummies(
        df["subscription_type"], prefix="sub"
    )
    contract_dummies = pd.get_dummies(
        df["contract_length"], prefix="contract"
    )
    tenure_dummies = pd.get_dummies(
        df["tenure_category"], prefix="tenuregroup"
    )
    age_dummies = pd.get_dummies(
        df["age_group"], prefix="agegroup"
    )
    spend_dummies = pd.get_dummies(
        df["spend_category"], prefix="spendcategory"
    )

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
    # 3️⃣ DROP ORIGINAL CATEGORICAL COLUMNS
    # =====================================================
    df = df.drop(
        columns=[
            "subscription_type",
            "contract_length",
            "tenure_category",
            "age_group",
            "spend_category",
        ]
    )

    # =====================================================
    # 4️⃣ SCALE NUMERICAL FEATURES
    # =====================================================
    excluded_cols = {"customerid", "churn", "gender"}
    features_to_scale = [
        col
        for col in df.columns
        if col not in excluded_cols and np.issubdtype(df[col].dtype, np.number)
    ]

    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    print(f"Scaled {len(features_to_scale)} numerical features")

    # =====================================================
    # 5️⃣ SAVE OUTPUT
    # =====================================================
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / name
    df.to_csv(output_file, index=False)

    return df
