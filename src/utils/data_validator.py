from typing import List, Tuple

import great_expectations as gx
import pandas as pd


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset
    using Great Expectations (compatible with all versions).

    Returns:
        success (bool): Whether all critical checks passed
        failed_expectations (List[str]): List of failed expectation types
    """

    print("🔍 Starting data validation with Great Expectations")

    # Use PandasDataset directly - compatible with all GE versions
    from great_expectations.dataset import PandasDataset

    validator = PandasDataset(df)

    # ================= SCHEMA VALIDATION =================
    print("Validating schema and required columns...")

    # Core identifiers and target
    validator.expect_column_to_exist("CustomerID")
    validator.expect_column_values_to_not_be_null("CustomerID")
    validator.expect_column_to_exist("Churn")

    # Demographics and usage
    validator.expect_column_to_exist("Age")
    validator.expect_column_to_exist("Gender")
    validator.expect_column_to_exist("Tenure")
    validator.expect_column_to_exist("Usage Frequency")
    validator.expect_column_to_exist("Support Calls")
    validator.expect_column_to_exist("Payment Delay")

    # Product / contract fields
    validator.expect_column_to_exist("Subscription Type")
    validator.expect_column_to_exist("Contract Length")

    # Financial / interaction fields
    validator.expect_column_to_exist("Total Spend")
    validator.expect_column_to_exist("Last Interaction")

    # ================= BUSINESS LOGIC =================
    print("Validating business logic constraints...")

    # value sets for categorical fields in train.csv
    validator.expect_column_values_to_be_in_set("Gender", ["Male", "Female"])
    validator.expect_column_values_to_be_in_set(
        "Subscription Type",
        ["Basic", "Standard", "Premium"],
    )
    validator.expect_column_values_to_be_in_set(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"],
    )
    # Churn should be binary (allow numeric or string forms)
    validator.expect_column_values_to_be_in_set("Churn", [0, 1, "0", "1"])

    # ================= NUMERIC RANGE =================
    print("Validating numeric ranges...")

    validator.expect_column_values_to_be_between("Tenure", min_value=0)
    validator.expect_column_values_to_be_between("Total Spend", min_value=0)
    validator.expect_column_values_to_be_between("Age", min_value=18, max_value=120)
    validator.expect_column_values_to_be_between(
        "Usage Frequency", min_value=0, max_value=31
    )
    validator.expect_column_values_to_be_between(
        "Support Calls", min_value=0, max_value=365
    )
    validator.expect_column_values_to_be_between(
        "Payment Delay", min_value=0, max_value=365
    )
    validator.expect_column_values_to_be_between(
        "Last Interaction", min_value=0, max_value=365
    )

    # ================= STATISTICAL CONSTRAINTS =================
    print("Validating statistical properties...")

    validator.expect_column_values_to_be_between("Tenure", min_value=0, max_value=120)
    validator.expect_column_values_to_not_be_null("Tenure")
    validator.expect_column_values_to_not_be_null("Total Spend")
    validator.expect_column_values_to_not_be_null("Age")
    validator.expect_column_mean_to_be_between(
        "Usage Frequency", min_value=1, max_value=20
    )

    # ================= DATA CONSISTENCY =================
    print("Validating data consistency...")

    # Simple consistency check: total spend should generally be >= support calls (numeric)
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="Total Spend",
        column_B="Support Calls",
        or_equal=True,
        mostly=0.95,
    )

    # 6. Run validation
    print("Running complete validation rules on the data...")
    results = validator.validate()

    # 7. Process results
    failed_expectations: List[str] = []  # store failed expectations rules here.

    # Handle both dict and ValidationResult object
    if hasattr(results, "results"):
        result_list = results.results
        success = results.success
    else:
        result_list = results.get("results", [])
        success = results.get("success", True)

    for r in result_list:
        if isinstance(r, dict):
            if not r.get("success", True):
                exp_type = r.get("expectation_config", {}).get(
                    "expectation_type", "unknown"
                )
                failed_expectations.append(exp_type)
        elif hasattr(r, "success") and not r.success:
            exp_type = getattr(r.expectation_config, "expectation_type", "unknown")
            failed_expectations.append(exp_type)

    total_checks = len(result_list)
    passed_checks = total_checks - len(failed_expectations)

    if success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks}")
    else:
        print(f"⚠️ Data validation FAILED: {passed_checks}/{total_checks}")
        print(f"   Failed expectations: {failed_expectations}")

    return success, failed_expectations
