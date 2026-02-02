import pandas as pd


def build_feature(df):
    print("\nOriginal Features Shape:", df.shape)

    # 1. Customer Lifetime Value (CLV) - based on spend and tenure
    df["clv"] = df["total_spend"] * (df["tenure"] / 12)

    # 2. Support Efficiency - Support calls relative to tenure
    # how much support calls need for per month. more calls mean less efficiency and likely to churn.
    df["support_efficiency"] = df["support_calls"] / (df["tenure"] + 1)

    # 3. Payment Reliability => inverse of payment delay
    df["payment_reliability"] = 1 / (df["payment_delay"] + 1)

    # 4. Usage Score - normalized usage frequency
    df["usage_score"] = df["usage_frequency"] / df["usage_frequency"].max() * 100

    # 5. Engagement Index - combines usage and support interaction
    df["engagement_index"] = (df["usage_frequency"] + df["support_calls"]) / 2

    # 6. Spend per Interaction - total spend relative to last interaction
    df["spend_per_interaction"] = df["total_spend"] / (df["last_interaction"] + 1)

    # 7. Risk Score - composite metric indicating churn risk
    df["risk_score"] = (
        (df["payment_delay"] / df["payment_delay"].max()) * 0.3
        + (1 - df["usage_frequency"] / df["usage_frequency"].max()) * 0.3
        + (df["tenure"] / df["tenure"].max())
        * (-0.2)  # Negative because longer tenure = lower risk
        + (df["support_calls"] / df["support_calls"].max()) * 0.2
    )

    # 8. tenure Category - binned tenure
    df["tenure_category"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60],
        labels=["0-12M", "12-24M", "24-36M", "36-48M", "48M+"],
    )

    # 9. age Group - binned age
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["18-30", "30-40", "40-50", "50-60", "60+"],
    )

    # 10. Spend Category - binned total spend
    df["spend_category"] = pd.cut(
        df["total_spend"], bins=3, labels=["Low", "Medium", "High"]
    )

    print("\nNew Features Shape:", df.shape)

    return df
