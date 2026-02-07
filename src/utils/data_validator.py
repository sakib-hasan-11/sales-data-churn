from typing import List, Tuple
import pandas as pd
import great_expectations as gx


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("🔍 Starting data validation with Great Expectations")

    # ================= GX CONTEXT =================
    context = gx.get_context()

    datasource_name = "pandas_ci_ds"
    asset_name = "runtime_df_asset"
    suite_name = "churn_suite"

    # ---- datasource ----
    try:
        datasource = context.data_sources.get(datasource_name)
    except Exception:
        datasource = context.data_sources.add_pandas(datasource_name)

    # ---- asset ----
    try:
        asset = datasource.get_asset(asset_name)
    except Exception:
        asset = datasource.add_dataframe_asset(asset_name)

    # ---- build validator (OUTSIDE try/except) ----
    batch_request = asset.build_batch_request(options={"dataframe": df})

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # ================= SCHEMA =================
    print("Validating schema...")

    required_cols = [
        "CustomerID","Churn","Age","Gender","Tenure",
        "Usage Frequency","Support Calls","Payment Delay",
        "Subscription Type","Contract Length",
        "Total Spend","Last Interaction"
    ]

    for col in required_cols:
        validator.expect_column_to_exist(col)

    validator.expect_column_values_to_not_be_null("CustomerID")

    # ================= BUSINESS =================
    print("Validating business logic...")

    validator.expect_column_values_to_be_in_set("Gender", ["Male", "Female"])
    validator.expect_column_values_to_be_in_set(
        "Subscription Type", ["Basic", "Standard", "Premium"]
    )
    validator.expect_column_values_to_be_in_set(
        "Contract Length", ["Monthly", "Quarterly", "Annual"]
    )
    validator.expect_column_values_to_be_in_set("Churn", [0, 1, "0", "1"])

    # ================= NUMERIC =================
    print("Validating numeric ranges...")

    validator.expect_column_values_to_be_between("Tenure", min_value=0)
    validator.expect_column_values_to_be_between("Total Spend", min_value=0)
    validator.expect_column_values_to_be_between("Age", min_value=18, max_value=120)
    validator.expect_column_values_to_be_between("Usage Frequency", min_value=0, max_value=31)
    validator.expect_column_values_to_be_between("Support Calls", min_value=0, max_value=365)
    validator.expect_column_values_to_be_between("Payment Delay", min_value=0, max_value=365)
    validator.expect_column_values_to_be_between("Last Interaction", min_value=0, max_value=365)

    # ================= STATS =================
    print("Validating statistics...")
    validator.expect_column_mean_to_be_between("Usage Frequency", min_value=1, max_value=20)

    # ================= CONSISTENCY =================
    print("Validating consistency...")
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="Total Spend",
        column_B="Support Calls",
        or_equal=True,
        mostly=0.95,
    )

    # ================= RUN =================
    print("Running GX validation...")
    results = validator.validate()

    failed_expectations: List[str] = []

    for r in results["results"]:
        if not r["success"]:
            failed_expectations.append(
                r["expectation_config"]["expectation_type"]
            )

    success = results["success"]

    total_checks = len(results["results"])
    passed_checks = total_checks - len(failed_expectations)

    if success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks}")
    else:
        print(f"⚠️ Data validation FAILED: {passed_checks}/{total_checks}")
        print(f"Failed expectations: {failed_expectations}")

    return success, failed_expectations
