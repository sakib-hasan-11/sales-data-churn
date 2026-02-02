from pathlib import Path
import pytest
from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.features.build_feature import build_feature
from src.features.feature_preprocess import preprocess_features

# =====================================================
# 1️ LOAD RAW DATA
# =====================================================


@pytest.fixture(scope="session")
def raw_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'tests' / "dummy_data.csv"

    df = load_data(data_path)
    return df



def test_raw_data_loaded(raw_data):
    assert raw_data is not None
    assert raw_data.shape[0] > 0
    assert "Churn" in raw_data.columns


# =====================================================
# 2️ RAW PREPROCESS
# =====================================================

@pytest.fixture(scope="session")
def processed_data(raw_data):
    df_processed = raw_preprocess(raw_data)
    return df_processed


def test_no_null_after_raw_preprocess(processed_data):
    assert processed_data.isnull().sum().sum() == 0


# =====================================================
# 3️ BUILD FEATURES
# =====================================================

@pytest.fixture(scope="session")
def feature_data(processed_data):
    df_features = build_feature(processed_data)
    return df_features


EXPECTED_FEATURES = [
    "clv",
    "support_efficiency",
    "payment_reliability",
    "usage_score",
    "engagement_index",
    "spend_per_interaction",
    "risk_score",
    "tenure_category",
    "age_group",
    "spend_category",
]



def test_feature_columns_present(feature_data):
    for col in EXPECTED_FEATURES:
        assert col in feature_data.columns


def test_no_null_in_features_except_categories(feature_data):
    allowed_nulls = {
        "tenure_category",
        "age_group",
        "spend_category",
    }

    cols_to_check = [
        c for c in feature_data.columns if c not in allowed_nulls
    ]

    assert feature_data[cols_to_check].isnull().sum().sum() == 0



# =====================================================
# 4️ FEATURE PREPROCESS (ONE-HOT + DROP + SAVE)
# =====================================================

@pytest.fixture(scope="module")
def preprocessed_feature_data(feature_data, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("preprocessed")

    df = preprocess_features(
        feature_data,
        output_dir,
        "preprocessed_features.csv"
    )

    return {
        "df": df,
        "output_dir": output_dir,
        "output_file": output_dir / "preprocessed_features.csv",
    }


def test_preprocessed_file_created(preprocessed_feature_data):
    assert preprocessed_feature_data["output_file"].exists()


def test_no_null_after_feature_preprocess(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    assert df.isnull().sum().sum() == 0


# =====================================================
# 5️ ONE-HOT ENCODING VALIDATION
# =====================================================

DROPPED_COLUMNS = [
    "subscription_type",
    "contract_length",
    "tenure_category",
    "age_group",
    "spend_category",
]

def test_target_column_present(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    assert "churn" in df.columns


def test_original_categorical_columns_removed(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    for col in DROPPED_COLUMNS:
        assert col not in df.columns

EXPECTED_PREFIXES = [
    "sub_",
    "contract_",
    "tenuregroup_",
    "agegroup_",
    "spendcategory_",
]



def test_dummy_columns_created(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    columns = df.columns.tolist()

    for prefix in EXPECTED_PREFIXES:
        assert any(
            col.startswith(prefix) for col in columns
        ), f"No column found with prefix {prefix}"


def test_one_hot_values_are_binary(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]

    dummy_columns = [
        col for col in df.columns
        if col.startswith(
            ("sub_","contract_","tenuregroup_","agegroup_","spendcategory_")
        )
    ]

    for col in dummy_columns:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0, 1}), f"{col} has non-binary values"


def test_one_hot_sum_rule(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]

    sub_cols = [c for c in df.columns if c.startswith("Sub_")]
    assert (df[sub_cols].sum(axis=1) <= 1).all()

