"""
Streamlit App - Sales Data Churn Project Architecture Viewer
Displays complete project architecture with flowcharts and function details
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Sales Churn Project Architecture",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        padding: 1rem 0;
        border-bottom: 3px solid #ff7f0e;
    }
    .function-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .pipeline-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .metric-box {
        background-color: #e1f5ff;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #1f77b4;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Project metadata
PROJECT_INFO = {
    "name": "Sales Data Churn Prediction",
    "description": "End-to-End Machine Learning Pipeline for Customer Churn Prediction",
    "modules": 5,
    "functions": 12,
    "pipeline_stages": 5,
}

# Function definitions for each module
FUNCTIONS_DATA = {
    "data_processing": {
        "load.py": [
            {
                "name": "load_data",
                "inputs": ["file_path: str/Path"],
                "outputs": ["pd.DataFrame"],
                "description": "Loads raw data from CSV file (train/test/holdout)",
                "arguments": {"file_path": "Path to the CSV file to load"},
                "returns": "DataFrame containing the loaded dataset with all features",
            }
        ],
        "preprocess.py": [
            {
                "name": "raw_preprocess",
                "inputs": [
                    "df: pd.DataFrame",
                    "strategy: str = 'auto'",
                    "custom_fill: Optional[Dict[str, Any]] = None",
                ],
                "outputs": ["pd.DataFrame"],
                "description": "Cleans column names and fills missing values using specified strategy",
                "arguments": {
                    "df": "Input DataFrame to preprocess",
                    "strategy": "Fill strategy: 'auto', 'constant', 'median', 'mode' (default: 'auto')",
                    "custom_fill": "Dictionary mapping column names to custom fill values (optional)",
                },
                "returns": "DataFrame with cleaned column names and filled missing values",
            }
        ],
    },
    "features": {
        "build_feature.py": [
            {
                "name": "build_feature",
                "inputs": ["df: pd.DataFrame"],
                "outputs": ["pd.DataFrame"],
                "description": "Creates 10 new engineered features: CLV, support efficiency, payment reliability, usage score, engagement index, spend per interaction, risk score, tenure category, age group, spend category",
                "arguments": {"df": "Input DataFrame with raw features"},
                "returns": "DataFrame with original + 10 new engineered features",
            }
        ],
        "feature_preprocess.py": [
            {
                "name": "preprocess_features",
                "inputs": ["df: pd.DataFrame", "output_path: str/Path", "name: str"],
                "outputs": ["pd.DataFrame"],
                "description": "Encodes categorical variables (label encoding + one-hot encoding), scales numerical features using StandardScaler, and saves processed data",
                "arguments": {
                    "df": "Input DataFrame with engineered features",
                    "output_path": "Directory path to save processed data",
                    "name": "Output filename for processed data",
                },
                "returns": "Fully preprocessed DataFrame ready for modeling",
            }
        ],
    },
    "training": {
        "optuna_tuning.py": [
            {
                "name": "optimize_xgboost_hyperparameters",
                "inputs": [
                    "train_data: pd.DataFrame",
                    "test_data: pd.DataFrame",
                    "target_col: str",
                    "n_trials: int = 100",
                    "optimize_metric: str = 'recall'",
                ],
                "outputs": ["Dict[str, Any]"],
                "description": "Uses Optuna to optimize XGBoost hyperparameters for specified metric",
                "arguments": {
                    "train_data": "Training DataFrame with features and target",
                    "test_data": "Test DataFrame with features and target",
                    "target_col": "Name of the target column",
                    "n_trials": "Number of Optuna optimization trials (default: 100)",
                    "optimize_metric": "Metric to optimize: 'recall', 'precision', 'f1', 'auc' (default: 'recall')",
                },
                "returns": "Dictionary with best_params, best_score, optimized_metric, all_metrics, study, final_model",
            }
        ],
        "mlflow_training.py": [
            {
                "name": "setup_mlflow_tracking",
                "inputs": ["mlflow_tracking_uri: str"],
                "outputs": ["str"],
                "description": "Sets up MLflow tracking for local/CI/production environments",
                "arguments": {
                    "mlflow_tracking_uri": "URI for MLflow tracking (file path or remote server)"
                },
                "returns": "Configured MLflow tracking URI",
            },
            {
                "name": "train_xgboost_with_mlflow",
                "inputs": [
                    "train_data: pd.DataFrame",
                    "test_data: pd.DataFrame",
                    "target_col: str",
                    "threshold_value: float",
                    "experiment_name: str = 'XGBoost_Threshold_Experiment'",
                    "n_optuna_trials: int = 100",
                    "n_runs: int = 5",
                    "mlflow_tracking_uri: str = './mlruns'",
                    "model_save_dir: str = 'models'",
                ],
                "outputs": ["Dict[str, Any]"],
                "description": "Trains XGBoost model with MLflow experiment tracking for specified threshold value, runs multiple experiments with best Optuna parameters",
                "arguments": {
                    "train_data": "Training DataFrame with features and target",
                    "test_data": "Test DataFrame with features and target",
                    "target_col": "Name of the target column",
                    "threshold_value": "Classification threshold (e.g., 0.5)",
                    "experiment_name": "Name for MLflow experiment (default: 'XGBoost_Threshold_Experiment')",
                    "n_optuna_trials": "Number of Optuna trials (default: 100)",
                    "n_runs": "Number of runs for this threshold (default: 5)",
                    "mlflow_tracking_uri": "MLflow tracking URI (default: './mlruns')",
                    "model_save_dir": "Directory to save models (default: 'models')",
                },
                "returns": "Dictionary with experiment_results, best_params, mlflow_tracking_uri",
            },
        ],
        "evaluation.py": [
            {
                "name": "save_model_from_run",
                "inputs": [
                    "run_id: str",
                    "mlflow_tracking_uri: str = './mlruns'",
                    "model_save_dir: str = './models'",
                ],
                "outputs": ["dict"],
                "description": "Saves trained model from MLflow run to production folder with metrics",
                "arguments": {
                    "run_id": "MLflow run ID to load model from",
                    "mlflow_tracking_uri": "Path to mlruns folder (default: './mlruns')",
                    "model_save_dir": "Directory to save production model (default: './models')",
                },
                "returns": "Dictionary with model_path, recall, threshold, run_id",
            }
        ],
    },
    "utils": {
        "data_validator.py": [
            {
                "name": "validate_data",
                "inputs": ["df: pd.DataFrame"],
                "outputs": ["Tuple[bool, List[str]]"],
                "description": "Validates data using Great Expectations: schema validation, business logic constraints, numeric ranges, statistical properties, and data consistency checks",
                "arguments": {"df": "DataFrame to validate against expectations"},
                "returns": "Tuple of (success: bool, failed_expectations: List[str])",
            }
        ]
    },
}

# Pipeline flow
PIPELINE_STAGES = [
    {
        "stage": "1. Data Loading",
        "module": "data_processing.load",
        "functions": ["load_data"],
        "description": "Load raw CSV data from data/raw/ folder",
        "icon": "📥",
    },
    {
        "stage": "2. Data Validation",
        "module": "utils.data_validator",
        "functions": ["validate_data"],
        "description": "Validate data quality using Great Expectations",
        "icon": "✅",
    },
    {
        "stage": "3. Data Preprocessing",
        "module": "data_processing.preprocess",
        "functions": ["raw_preprocess"],
        "description": "Clean column names and handle missing values",
        "icon": "🧹",
    },
    {
        "stage": "4. Feature Engineering",
        "module": "features.build_feature",
        "functions": ["build_feature"],
        "description": "Create 10 engineered features (CLV, risk score, etc.)",
        "icon": "🔧",
    },
    {
        "stage": "5. Feature Preprocessing",
        "module": "features.feature_preprocess",
        "functions": ["preprocess_features"],
        "description": "Encode categories and scale numerical features",
        "icon": "📊",
    },
    {
        "stage": "6. Hyperparameter Tuning",
        "module": "training.optuna_tuning",
        "functions": ["optimize_xgboost_hyperparameters"],
        "description": "Optimize XGBoost hyperparameters using Optuna",
        "icon": "⚙️",
    },
    {
        "stage": "7. Model Training & Tracking",
        "module": "training.mlflow_training",
        "functions": ["setup_mlflow_tracking", "train_xgboost_with_mlflow"],
        "description": "Train XGBoost models with MLflow experiment tracking",
        "icon": "🤖",
    },
    {
        "stage": "8. Model Evaluation & Saving",
        "module": "training.evaluation",
        "functions": ["save_model_from_run"],
        "description": "Save best model to production folder",
        "icon": "💾",
    },
]


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🏗️ Sales Data Churn Project Architecture</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    page = st.sidebar.radio(
        "Select View",
        [
            "🏠 Overview",
            "🔄 Pipeline Flow",
            "📦 Module Explorer",
            "🎯 Function Details",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Project Stats")
    st.sidebar.metric("Total Modules", PROJECT_INFO["modules"])
    st.sidebar.metric("Total Functions", PROJECT_INFO["functions"])
    st.sidebar.metric("Pipeline Stages", PROJECT_INFO["pipeline_stages"])

    # Page routing
    if page == "🏠 Overview":
        show_overview()
    elif page == "🔄 Pipeline Flow":
        show_pipeline_flow()
    elif page == "📦 Module Explorer":
        show_module_explorer()
    elif page == "🎯 Function Details":
        show_function_details()


def show_overview():
    """Display project overview"""
    st.markdown(
        '<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        ### {PROJECT_INFO["name"]}
        
        {PROJECT_INFO["description"]}
        
        This project implements a complete machine learning pipeline for predicting customer churn
        using XGBoost with advanced features like:
        
        - 🔍 **Data Validation** using Great Expectations
        - 🔧 **Feature Engineering** with 10 derived features
        - ⚙️ **Hyperparameter Tuning** using Optuna
        - 📊 **Experiment Tracking** with MLflow
        - 🎯 **Model Optimization** for recall maximization
        """)

    with col2:
        st.markdown("### 🗂️ Project Structure")
        st.code(
            """
src/
├── data_processing/
│   ├── load.py
│   └── preprocess.py
├── features/
│   ├── build_feature.py
│   └── feature_preprocess.py
├── training/
│   ├── optuna_tuning.py
│   ├── mlflow_training.py
│   └── evaluation.py
└── utils/
    └── data_validator.py
        """,
            language="text",
        )

    # Architecture Diagram
    st.markdown("### 🏗️ High-Level Architecture")

    st.markdown("""
    ```mermaid
    graph TD
        A[Raw Data CSV] --> B[Data Loading]
        B --> C[Data Validation]
        C --> D[Data Preprocessing]
        D --> E[Feature Engineering]
        E --> F[Feature Preprocessing]
        F --> G[Hyperparameter Tuning]
        G --> H[Model Training]
        H --> I[MLflow Tracking]
        I --> J[Model Evaluation]
        J --> K[Production Model]
        
        style A fill:#e1f5ff
        style K fill:#d4edda
        style G fill:#fff3cd
        style H fill:#fff3cd
        style I fill:#fff3cd
    ```
    """)


def show_pipeline_flow():
    """Display pipeline flow with detailed stages"""
    st.markdown(
        '<h2 class="section-header">🔄 ML Pipeline Flow</h2>', unsafe_allow_html=True
    )

    st.markdown("""
    This section shows the complete machine learning pipeline from raw data to production model.
    Each stage is executed sequentially with outputs feeding into the next stage.
    """)

    # Display pipeline stages
    for i, stage in enumerate(PIPELINE_STAGES):
        with st.expander(
            f"{stage['icon']} {stage['stage']}: {stage['description']}", expanded=False
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Module:**")
                st.code(stage["module"])
                st.markdown("**Functions:**")
                for func in stage["functions"]:
                    st.code(f"• {func}()")

            with col2:
                st.markdown("**Description:**")
                st.info(stage["description"])

                # Show connections
                if i < len(PIPELINE_STAGES) - 1:
                    st.markdown(
                        f"**⬇️ Output flows to:** {PIPELINE_STAGES[i + 1]['stage']}"
                    )

    # Flowchart visualization
    st.markdown("---")
    st.markdown("### 📊 Detailed Pipeline Flowchart")

    st.markdown("""
    ```mermaid
    graph LR
        A[📥 Raw CSV Data] --> B[load_data]
        B --> C[validate_data]
        C --> D{Valid?}
        D -->|Yes| E[raw_preprocess]
        D -->|No| Z[Error Report]
        E --> F[build_feature]
        F --> G[preprocess_features]
        G --> H[optimize_xgboost_hyperparameters]
        H --> I[setup_mlflow_tracking]
        I --> J[train_xgboost_with_mlflow]
        J --> K[save_model_from_run]
        K --> L[💾 Production Model]
        
        style A fill:#e1f5ff
        style L fill:#d4edda
        style Z fill:#f8d7da
        style H fill:#fff3cd
        style J fill:#fff3cd
    ```
    """)


def show_module_explorer():
    """Display module explorer with buttons"""
    st.markdown(
        '<h2 class="section-header">📦 Module Explorer</h2>', unsafe_allow_html=True
    )

    st.markdown(
        "Explore each module and its functions. Click on a module to see its functions."
    )

    # Module selector
    modules = list(FUNCTIONS_DATA.keys())

    for module_name in modules:
        st.markdown(f"### 📁 {module_name}")

        module_data = FUNCTIONS_DATA[module_name]
        files = list(module_data.keys())

        cols = st.columns(len(files))

        for idx, file_name in enumerate(files):
            with cols[idx]:
                if st.button(
                    f"📄 {file_name}",
                    key=f"btn_{module_name}_{file_name}",
                    use_container_width=True,
                ):
                    st.session_state["selected_file"] = (module_name, file_name)

        # Show selected file functions
        if "selected_file" in st.session_state:
            sel_module, sel_file = st.session_state["selected_file"]
            if sel_module == module_name:
                st.markdown(f"#### Functions in `{sel_file}`:")

                for func_data in module_data[sel_file]:
                    with st.container():
                        st.markdown(f"##### 🔹 `{func_data['name']}()`")
                        st.markdown(func_data["description"])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📥 Inputs:**")
                            for inp in func_data["inputs"]:
                                st.code(inp)

                        with col2:
                            st.markdown("**📤 Output:**")
                            for out in func_data["outputs"]:
                                st.code(out)

                        st.markdown("---")

        st.markdown("---")


def show_function_details():
    """Display detailed function information"""
    st.markdown(
        '<h2 class="section-header">🎯 Function Details</h2>', unsafe_allow_html=True
    )

    st.markdown(
        "Detailed view of all functions with their inputs, arguments, and outputs."
    )

    # Module selector
    module_name = st.selectbox(
        "Select Module", list(FUNCTIONS_DATA.keys()), format_func=lambda x: f"📁 {x}"
    )

    if module_name:
        module_data = FUNCTIONS_DATA[module_name]
        file_name = st.selectbox(
            "Select File", list(module_data.keys()), format_func=lambda x: f"📄 {x}"
        )

        if file_name:
            st.markdown(f"### Functions in `{module_name}/{file_name}`")

            for func_data in module_data[file_name]:
                st.markdown(f"#### 🔹 `{func_data['name']}()`")

                # Function description
                st.info(func_data["description"])

                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📥 Inputs", "⚙️ Arguments", "📤 Returns", "💡 Example"]
                )

                with tab1:
                    st.markdown("**Input Parameters:**")
                    for inp in func_data["inputs"]:
                        st.code(inp, language="python")

                with tab2:
                    st.markdown("**Argument Details:**")
                    for arg_name, arg_desc in func_data["arguments"].items():
                        st.markdown(f"**`{arg_name}`**")
                        st.markdown(f"  - {arg_desc}")

                with tab3:
                    st.markdown("**Return Value:**")
                    st.code(func_data["returns"], language="python")
                    for out in func_data["outputs"]:
                        st.code(out, language="python")

                with tab4:
                    st.markdown("**Usage Example:**")

                    # Generate example based on function
                    if func_data["name"] == "load_data":
                        example = """
import pandas as pd
from src.data_processing.load import load_data

# Load training data
df = load_data('data/raw/train.csv')
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
                        """
                    elif func_data["name"] == "raw_preprocess":
                        example = """
from src.data_processing.preprocess import raw_preprocess

# Preprocess with auto strategy
df_clean = raw_preprocess(df, strategy='auto')

# Preprocess with custom fills
custom_fills = {'age': 30, 'tenure': 0}
df_clean = raw_preprocess(df, strategy='auto', custom_fill=custom_fills)
                        """
                    elif func_data["name"] == "build_feature":
                        example = """
from src.features.build_feature import build_feature

# Add engineered features
df_engineered = build_feature(df)
print(f"Original features: {df.shape[1]}")
print(f"After engineering: {df_engineered.shape[1]}")
                        """
                    elif func_data["name"] == "preprocess_features":
                        example = """
from src.features.feature_preprocess import preprocess_features

# Encode and scale features
df_processed = preprocess_features(
    df=df_engineered,
    output_path='data/processed',
    name='train_processed.csv'
)
                        """
                    elif func_data["name"] == "optimize_xgboost_hyperparameters":
                        example = """
from src.training.optuna_tuning import optimize_xgboost_hyperparameters

results = optimize_xgboost_hyperparameters(
    train_data=train_df,
    test_data=test_df,
    target_col='churn',
    n_trials=100,
    optimize_metric='recall'
)

print("Best parameters:", results['best_params'])
print("Best recall:", results['best_score'])
                        """
                    elif func_data["name"] == "train_xgboost_with_mlflow":
                        example = """
from src.training.mlflow_training import train_xgboost_with_mlflow

results = train_xgboost_with_mlflow(
    train_data=train_df,
    test_data=test_df,
    target_col='churn',
    threshold_value=0.5,
    experiment_name='Churn_Prediction_0.5',
    n_optuna_trials=100,
    n_runs=5
)

print("Best recall:", results['experiment_results']['best_recall'])
                        """
                    elif func_data["name"] == "validate_data":
                        example = """
from src.utils.data_validator import validate_data

success, failed_checks = validate_data(df)

if success:
    print("✅ Data validation passed!")
else:
    print("❌ Validation failed:")
    print(f"Failed checks: {failed_checks}")
                        """
                    else:
                        example = f"""
from src.{module_name}.{file_name.replace(".py", "")} import {func_data["name"]}

# Call the function
result = {func_data["name"]}(<arguments>)
                        """

                    st.code(example, language="python")

                st.markdown("---")


# Run the app
if __name__ == "__main__":
    main()
