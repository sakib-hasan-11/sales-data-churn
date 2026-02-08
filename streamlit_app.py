"""
Streamlit Frontend for Churn Prediction API
Interactive UI for single and batch predictions
"""

import os
import time
from io import StringIO

import pandas as pd
import requests
import streamlit as st

# ============================================================================
# Configuration
# ============================================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")
APP_TITLE = "🔮 Customer Churn Prediction"
APP_ICON = "🔮"

# ============================================================================
# Helper Functions
# ============================================================================


def check_api_health():
    """Check if API is healthy and reachable"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def predict_single(customer_data):
    """Make single prediction via API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=customer_data, timeout=10)

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "error": f"API Error: {response.status_code}",
                "details": response.text,
            }
    except Exception as e:
        return False, {"error": str(e)}


def predict_batch(customers_data):
    """Make batch prediction via API"""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch", json={"customers": customers_data}, timeout=30
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "error": f"API Error: {response.status_code}",
                "details": response.text,
            }
    except Exception as e:
        return False, {"error": str(e)}


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# Sidebar - API Status and Model Info
# ============================================================================

with st.sidebar:
    st.header("🔧 System Status")

    # Check API health
    is_healthy, health_data = check_api_health()

    if is_healthy:
        st.success("✅ API is Online")

        if "model_loaded" in health_data and health_data["model_loaded"]:
            st.success("✅ Model Loaded")
        else:
            st.warning("⚠️ Model Not Loaded")

        # Get model info
        model_info = get_model_info()
        if model_info:
            st.info(f"**Model Version:** {model_info.get('model_version', 'Unknown')}")
            st.info(f"**Source:** {model_info.get('model_source', 'Unknown')}")
    else:
        st.error("❌ API is Offline")
        st.error(f"Error: {health_data.get('error', 'Unknown')}")
        st.warning(f"Trying to connect to: {API_URL}")

    st.divider()

    # API Configuration
    st.header("⚙️ Configuration")
    new_api_url = st.text_input("API URL", value=API_URL, key="api_url_input")
    if st.button("Update API URL"):
        API_URL = new_api_url
        st.rerun()

# ============================================================================
# Main Application
# ============================================================================

st.markdown(
    f'<div class="main-header">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True
)

# Tab selection
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction"])

# ============================================================================
# Tab 1: Single Prediction
# ============================================================================

with tab1:
    st.header("Single Customer Prediction")
    st.write("Enter customer details to predict churn probability")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Basic Information")
        customer_id = st.text_input("Customer ID", value="CUST001", key="single_id")
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35, key="single_age"
        )
        gender = st.selectbox("Gender", ["Male", "Female"], key="single_gender")
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=120, value=24, key="single_tenure"
        )

    with col2:
        st.subheader("📞 Engagement Metrics")
        usage_frequency = st.number_input(
            "Usage Frequency", min_value=0, max_value=100, value=15, key="single_usage"
        )
        support_calls = st.number_input(
            "Support Calls", min_value=0, max_value=50, value=3, key="single_support"
        )
        payment_delay = st.number_input(
            "Payment Delay (days)",
            min_value=0,
            max_value=90,
            value=5,
            key="single_delay",
        )
        last_interaction = st.number_input(
            "Last Interaction (days)",
            min_value=0,
            max_value=365,
            value=10,
            key="single_interaction",
        )

    with col3:
        st.subheader("💰 Subscription Details")
        subscription_type = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"],
            index=2,
            key="single_subscription",
        )
        contract_length = st.selectbox(
            "Contract Length",
            ["Monthly", "Quarterly", "Annual"],
            index=2,
            key="single_contract",
        )
        total_spend = st.number_input(
            "Total Spend ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1250.50,
            step=10.0,
            key="single_spend",
        )

    st.divider()

    # Predict button
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        if not is_healthy:
            st.error("❌ Cannot make prediction - API is offline")
        else:
            with st.spinner("Making prediction..."):
                customer_data = {
                    "customerid": customer_id,
                    "age": age,
                    "gender": gender,
                    "tenure": tenure,
                    "usage_frequency": usage_frequency,
                    "support_calls": support_calls,
                    "payment_delay": payment_delay,
                    "subscription_type": subscription_type,
                    "contract_length": contract_length,
                    "total_spend": total_spend,
                    "last_interaction": last_interaction,
                }

                success, result = predict_single(customer_data)

                if success:
                    st.success("✅ Prediction completed successfully!")

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        churn_prob = result.get("churn_probability", 0) * 100
                        st.metric(
                            label="Churn Probability",
                            value=f"{churn_prob:.1f}%",
                            delta=None,
                        )

                    with col2:
                        will_churn = result.get("will_churn", False)
                        st.metric(
                            label="Prediction",
                            value="WILL CHURN" if will_churn else "WILL STAY",
                            delta=None,
                        )

                    with col3:
                        risk_level = result.get("risk_level", "Unknown")
                        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(
                            risk_level, "⚪"
                        )
                        st.metric(
                            label="Risk Level",
                            value=f"{risk_color} {risk_level}",
                            delta=None,
                        )

                    # Show full response
                    with st.expander("📋 View Full Response"):
                        st.json(result)
                else:
                    st.error(
                        f"❌ Prediction failed: {result.get('error', 'Unknown error')}"
                    )
                    if "details" in result:
                        st.error(f"Details: {result['details']}")

# ============================================================================
# Tab 2: Batch Prediction
# ============================================================================

with tab2:
    st.header("Batch Customer Prediction")
    st.write("Upload a CSV file with customer data for batch predictions")

    # Download sample CSV template
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Upload Customer Data")

        # Show required columns
        with st.expander("ℹ️ Required CSV Columns"):
            st.code("""
customerid, age, gender, tenure, usage_frequency, 
support_calls, payment_delay, subscription_type, 
contract_length, total_spend, last_interaction
            """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with customer data",
        )

    with col2:
        st.subheader("📤 Download Sample Template")

        # Create sample data
        sample_data = pd.DataFrame(
            [
                {
                    "customerid": "CUST001",
                    "age": 35,
                    "gender": "Male",
                    "tenure": 24,
                    "usage_frequency": 15,
                    "support_calls": 3,
                    "payment_delay": 5,
                    "subscription_type": "Premium",
                    "contract_length": "Annual",
                    "total_spend": 1250.50,
                    "last_interaction": 10,
                },
                {
                    "customerid": "CUST002",
                    "age": 42,
                    "gender": "Female",
                    "tenure": 36,
                    "usage_frequency": 8,
                    "support_calls": 5,
                    "payment_delay": 15,
                    "subscription_type": "Basic",
                    "contract_length": "Monthly",
                    "total_spend": 450.00,
                    "last_interaction": 30,
                },
            ]
        )

        csv_buffer = StringIO()
        sample_data.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇️ Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="sample_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ File uploaded successfully! Found {len(df)} customers")

            # Show data preview
            with st.expander("👁️ Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)

            # Validate columns
            required_cols = [
                "customerid",
                "age",
                "gender",
                "tenure",
                "usage_frequency",
                "support_calls",
                "payment_delay",
                "subscription_type",
                "contract_length",
                "total_spend",
                "last_interaction",
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns present")

                # Predict button
                if st.button(
                    "🔮 Predict Batch", type="primary", use_container_width=True
                ):
                    if not is_healthy:
                        st.error("❌ Cannot make prediction - API is offline")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("Preparing data...")
                        progress_bar.progress(20)

                        # Convert dataframe to list of dicts
                        customers_list = df.to_dict("records")

                        status_text.text(
                            f"Sending {len(customers_list)} customers to API..."
                        )
                        progress_bar.progress(40)

                        # Make batch prediction
                        success, result = predict_batch(customers_list)
                        progress_bar.progress(80)

                        if success:
                            status_text.text("Processing results...")
                            progress_bar.progress(100)
                            time.sleep(0.5)

                            st.success(
                                f"✅ Batch prediction completed! Processed {result.get('total_customers', 0)} customers"
                            )

                            # Display summary statistics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "Total Customers", result.get("total_customers", 0)
                                )

                            with col2:
                                st.metric("Will Churn", result.get("churn_count", 0))

                            with col3:
                                st.metric("Will Stay", result.get("no_churn_count", 0))

                            with col4:
                                churn_rate = result.get("churn_rate", 0) * 100
                                st.metric("Churn Rate", f"{churn_rate:.1f}%")

                            # Risk distribution
                            st.subheader("📊 Risk Distribution")
                            risk_dist = result.get("risk_distribution", {})

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🟢 Low Risk", risk_dist.get("Low", 0))
                            with col2:
                                st.metric("🟡 Medium Risk", risk_dist.get("Medium", 0))
                            with col3:
                                st.metric("🔴 High Risk", risk_dist.get("High", 0))

                            # Results table
                            st.subheader("📋 Detailed Results")

                            predictions = result.get("predictions", [])
                            if predictions:
                                results_df = pd.DataFrame(predictions)

                                # Format probability as percentage
                                if "churn_probability" in results_df.columns:
                                    results_df["churn_probability_%"] = (
                                        results_df["churn_probability"] * 100
                                    ).round(2)

                                st.dataframe(results_df, use_container_width=True)

                                # Download results
                                csv_buffer = StringIO()
                                results_df.to_csv(csv_buffer, index=False)

                                st.download_button(
                                    label="⬇️ Download Results as CSV",
                                    data=csv_buffer.getvalue(),
                                    file_name="churn_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )

                            # Show full response
                            with st.expander("📋 View Full API Response"):
                                st.json(result)
                        else:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(
                                f"❌ Batch prediction failed: {result.get('error', 'Unknown error')}"
                            )
                            if "details" in result:
                                st.error(f"Details: {result['details']}")

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info(
                "💡 Make sure your CSV file is properly formatted with all required columns"
            )

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🔮 Customer Churn Prediction System | Built with Streamlit & FastAPI</p>
        <p style='font-size: 0.8rem;'>For support, contact your system administrator</p>
    </div>
""",
    unsafe_allow_html=True,
)
