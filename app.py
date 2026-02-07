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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a365d;
        text-align: center;
        padding: 2rem 0 1rem 0;
        letter-spacing: -0.5px;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0f766e;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .function-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .metric-box {
        background-color: #f8fafc;
        padding: 1.2rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
    }
    
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
    
    .icon {
        margin-right: 8px;
        color: #0f766e;
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
        "icon": "fa-solid fa-database",
    },
    {
        "stage": "2. Data Validation",
        "module": "utils.data_validator",
        "functions": ["validate_data"],
        "description": "Validate data quality using Great Expectations",
        "icon": "fa-solid fa-check-circle",
    },
    {
        "stage": "3. Data Preprocessing",
        "module": "data_processing.preprocess",
        "functions": ["raw_preprocess"],
        "description": "Clean column names and handle missing values",
        "icon": "fa-solid fa-broom",
    },
    {
        "stage": "4. Feature Engineering",
        "module": "features.build_feature",
        "functions": ["build_feature"],
        "description": "Create 10 engineered features (CLV, risk score, etc.)",
        "icon": "fa-solid fa-wrench",
    },
    {
        "stage": "5. Feature Preprocessing",
        "module": "features.feature_preprocess",
        "functions": ["preprocess_features"],
        "description": "Encode categories and scale numerical features",
        "icon": "fa-solid fa-chart-line",
    },
    {
        "stage": "6. Hyperparameter Tuning",
        "module": "training.optuna_tuning",
        "functions": ["optimize_xgboost_hyperparameters"],
        "description": "Optimize XGBoost hyperparameters using Optuna",
        "icon": "fa-solid fa-sliders",
    },
    {
        "stage": "7. Model Training & Tracking",
        "module": "training.mlflow_training",
        "functions": ["setup_mlflow_tracking", "train_xgboost_with_mlflow"],
        "description": "Train XGBoost models with MLflow experiment tracking",
        "icon": "fa-solid fa-brain",
    },
    {
        "stage": "8. Model Evaluation & Saving",
        "module": "training.evaluation",
        "functions": ["save_model_from_run"],
        "description": "Save best model to production folder",
        "icon": "fa-solid fa-floppy-disk",
    },
]


def main():
    # Header
    st.markdown(
        '<h1 class="main-header"><i class="fa-solid fa-diagram-project icon"></i>Sales Data Churn Project Architecture</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Select View",
        [
            "Overview",
            "Pipeline Flow",
            "Pipeline Scripts",
            "Module Explorer",
            "Function Details",
        ],
        format_func=lambda x: {
            "Overview": "  Overview",
            "Pipeline Flow": "  Pipeline Flow",
            "Pipeline Scripts": "  Pipeline Scripts",
            "Module Explorer": "  Module Explorer",
            "Function Details": "  Function Details",
        }[x],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Project Stats")
    st.sidebar.metric("Total Modules", PROJECT_INFO["modules"])
    st.sidebar.metric("Total Functions", PROJECT_INFO["functions"])
    st.sidebar.metric("Pipeline Stages", PROJECT_INFO["pipeline_stages"])

    # Page routing
    if page == "Overview":
        show_overview()
    elif page == "Pipeline Flow":
        show_pipeline_flow()
    elif page == "Pipeline Scripts":
        show_pipeline_scripts()
    elif page == "Module Explorer":
        show_module_explorer()
    elif page == "Function Details":
        show_function_details()


def show_overview():
    """Display project overview"""
    st.markdown(
        '<h2 class="section-header"><i class="fa-solid fa-house icon"></i>Project Overview</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        ### {PROJECT_INFO["name"]}
        
        {PROJECT_INFO["description"]}
        
        This project implements a complete machine learning pipeline for predicting customer churn
        using XGBoost with advanced features:
        """)

        features = [
            ("fa-shield-halved", "Data Validation", "using Great Expectations"),
            (
                "fa-wand-magic-sparkles",
                "Feature Engineering",
                "with 10 derived features",
            ),
            ("fa-sliders", "Hyperparameter Tuning", "using Optuna"),
            ("fa-chart-bar", "Experiment Tracking", "with MLflow"),
            ("fa-bullseye", "Model Optimization", "for recall maximization"),
        ]

        for icon, title, desc in features:
            st.markdown(
                f"""
            <div style="display: flex; align-items: center; margin: 12px 0;">
                <i class="fa-solid {icon}" style="font-size: 1.2rem; color: #0f766e; width: 35px;"></i>
                <span><strong>{title}</strong> {desc}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### Project Structure")
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
    """Display pipeline flow with interactive HTML flowchart"""
    st.markdown(
        '<h2 class="section-header"><i class="fa-solid fa-diagram-project icon"></i>ML Pipeline Flow</h2>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    Click on any stage in the pipeline below to see detailed information about data flow and transformations.
    """)

    # Create interactive HTML flowchart with professional design
    interactive_flowchart = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        .pipeline-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            background: #f8fafc;
            border-radius: 8px;
            margin: 1.5rem 0;
            font-family: 'Inter', sans-serif;
        }
        
        .pipeline-stage {
            position: relative;
            width: 90%;
            max-width: 900px;
            margin: 8px 0;
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            cursor: pointer;
            transition: all 0.25s ease;
            border-left: 4px solid #0f766e;
        }
        
        .pipeline-stage:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            transform: translateX(4px);
            border-left-color: #1a365d;
        }
        
        .pipeline-stage.active {
            background: #ffffff;
            border-left-color: #0f766e;
            box-shadow: 0 4px 16px rgba(15, 118, 110, 0.15);
            transform: translateX(0);
        }
        
        .stage-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        
        .stage-header-left {
            display: flex;
            align-items: center;
            flex: 1;
        }
        
        .stage-icon {
            font-size: 1.5rem;
            color: #0f766e;
            margin-right: 1rem;
            width: 40px;
            text-align: center;
        }
        
        .pipeline-stage.active .stage-icon {
            color: #1a365d;
        }
        
        .stage-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a365d;
            flex: 1;
        }
        
        .stage-description {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 0.25rem;
            line-height: 1.5;
        }
        
        .stage-arrow {
            font-size: 1.2rem;
            color: #cbd5e1;
            transition: all 0.25s ease;
        }
        
        .pipeline-stage:hover .stage-arrow {
            color: #94a3b8;
        }
        
        .pipeline-stage.active .stage-arrow {
            transform: rotate(90deg);
            color: #0f766e;
        }
        
        .stage-details {
            display: none;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
        }
        
        .pipeline-stage.active .stage-details {
            display: block;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(-8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .arrow-connector {
            width: 2px;
            height: 20px;
            background: linear-gradient(to bottom, #cbd5e1, #94a3b8);
            margin: 0 auto;
            position: relative;
        }
        
        .arrow-connector::after {
            content: '▼';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            color: #94a3b8;
            font-size: 0.8rem;
        }
        
        .detail-section {
            margin: 1rem 0;
        }
        
        .detail-label {
            font-weight: 600;
            color: #1a365d;
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
        }
        
        .detail-label i {
            margin-right: 0.5rem;
            color: #0f766e;
        }
        
        .detail-content {
            background: #f8fafc;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            color: #475569;
            border: 1px solid #e2e8f0;
        }
        
        .data-flow-box {
            background: #f0fdfa;
            padding: 1.25rem;
            border-radius: 6px;
            margin-top: 1rem;
            border: 1px solid #99f6e4;
        }
        
        .data-flow-title {
            font-weight: 600;
            color: #0f766e;
            margin-bottom: 1rem;
            font-size: 1rem;
            display: flex;
            align-items: center;
        }
        
        .data-flow-title i {
            margin-right: 0.5rem;
        }
        
        ul {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }
        
        li {
            margin: 0.4rem 0;
            color: #475569;
            line-height: 1.5;
        }
        
        .module-badge {
            display: inline-block;
            background: #e0f2fe;
            color: #0c4a6e;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .function-item {
            background: #ffffff;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            margin: 0.25rem 0;
            border-left: 3px solid #0f766e;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            color: #1e293b;
        }
    </style>
    
    <div class="pipeline-container">
    """

    # Define detailed data flow information for each stage
    data_flows = [
        {
            "input": "CSV files from data/raw/ (train.csv, test.csv, holdout.csv)",
            "process": "Reads CSV files using pandas and returns DataFrames",
            "output": "Raw DataFrame with all original columns",
            "columns": [
                "customer_id",
                "age",
                "tenure",
                "total_spend",
                "support_calls",
                "payment_delay",
                "churn",
            ],
        },
        {
            "input": "Raw DataFrame from load_data()",
            "process": "Validates data quality: checks schema, nulls, ranges, business rules using Great Expectations",
            "output": "Validation report (success/failure) + list of failed checks",
            "checks": [
                "Schema validation",
                "Null checks",
                "Numeric ranges",
                "Statistical properties",
                "Business constraints",
            ],
        },
        {
            "input": "Validated DataFrame",
            "process": "Cleans column names (lowercase, remove spaces), fills missing values using strategy",
            "output": "Clean DataFrame with standardized columns and no missing values",
            "transformations": [
                "Column name standardization",
                "Missing value imputation",
                "Data type validation",
            ],
        },
        {
            "input": "Preprocessed DataFrame",
            "process": "Creates 10 new features through mathematical transformations and aggregations",
            "output": "Enhanced DataFrame with original + 10 engineered features",
            "new_features": [
                "CLV",
                "Support Efficiency",
                "Payment Reliability",
                "Usage Score",
                "Engagement Index",
                "Spend per Interaction",
                "Risk Score",
                "Tenure Category",
                "Age Group",
                "Spend Category",
            ],
        },
        {
            "input": "DataFrame with engineered features",
            "process": "Encodes categorical variables (label + one-hot encoding), scales numerical features using StandardScaler",
            "output": "Model-ready DataFrame with encoded and scaled features, saved to data/processed/",
            "operations": [
                "Label encoding",
                "One-hot encoding",
                "StandardScaler normalization",
                "Data persistence",
            ],
        },
        {
            "input": "Processed train/test DataFrames",
            "process": "Runs Optuna trials to find best XGBoost hyperparameters optimizing for specified metric",
            "output": "Best hyperparameters dict + optimization metrics + Optuna study object",
            "parameters": [
                "learning_rate",
                "max_depth",
                "n_estimators",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
            ],
        },
        {
            "input": "Optimized hyperparameters + train/test data",
            "process": "Trains multiple XGBoost models, logs experiments to MLflow with metrics and artifacts",
            "output": "Trained models with metrics logged to MLflow, best model identified",
            "tracked": [
                "Recall",
                "Precision",
                "F1-score",
                "AUC-ROC",
                "Confusion Matrix",
                "Hyperparameters",
                "Model artifacts",
            ],
        },
        {
            "input": "MLflow run_id + metrics",
            "process": "Loads best model from MLflow, saves to production folder with metadata",
            "output": "Production-ready .pkl model file in models/ directory with performance metrics",
            "artifacts": [
                "model.pkl",
                "metrics.json",
                "preprocessing_config",
                "feature_importance",
            ],
        },
    ]

    for i, stage in enumerate(PIPELINE_STAGES):
        # Add stage
        stage_id = f"stage-{i}"
        interactive_flowchart += f"""
        <div class="pipeline-stage" id="{stage_id}" onclick="toggleStage('{stage_id}')">
            <div class="stage-header">
                <div class="stage-header-left">
                    <i class="{stage["icon"]} stage-icon"></i>
                    <div>
                        <div class="stage-title">{stage["stage"]}</div>
                        <div class="stage-description">{stage["description"]}</div>
                    </div>
                </div>
                <i class="fa-solid fa-chevron-right stage-arrow"></i>
            </div>
            
            <div class="stage-details">
                <div class="detail-section">
                    <div class="detail-label"><i class="fa-solid fa-cube"></i> Module</div>
                    <span class="module-badge">{stage["module"]}</span>
                </div>
                
                <div class="detail-section">
                    <div class="detail-label"><i class="fa-solid fa-code"></i> Functions</div>
                    {"".join([f'<div class="function-item">{func}()</div>' for func in stage["functions"]])}
                </div>
                
                <div class="data-flow-box">
                    <div class="data-flow-title"><i class="fa-solid fa-arrow-right-arrow-left"></i> Data Flow Through This Stage</div>
                    
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-arrow-down-to-line"></i> Input</div>
                        <div style="color: #475569; line-height: 1.6;">{data_flows[i]["input"]}</div>
                    </div>
                    
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-gear"></i> Processing</div>
                        <div style="color: #475569; line-height: 1.6;">{data_flows[i]["process"]}</div>
                    </div>
                    
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-arrow-up-from-line"></i> Output</div>
                        <div style="color: #475569; line-height: 1.6;">{data_flows[i]["output"]}</div>
                    </div>
        """

        # Add specific details based on stage
        if "columns" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-table-columns"></i> Output Columns</div>
                        <ul>
                            {"".join([f"<li>{col}</li>" for col in data_flows[i]["columns"]])}
                        </ul>
                    </div>
            """
        elif "checks" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-list-check"></i> Validation Checks</div>
                        <ul>
                            {"".join([f"<li>{check}</li>" for check in data_flows[i]["checks"]])}
                        </ul>
                    </div>
            """
        elif "transformations" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-shuffle"></i> Transformations</div>
                        <ul>
                            {"".join([f"<li>{trans}</li>" for trans in data_flows[i]["transformations"]])}
                        </ul>
                    </div>
            """
        elif "new_features" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-sparkles"></i> New Features Created</div>
                        <ul>
                            {"".join([f"<li>{feat}</li>" for feat in data_flows[i]["new_features"]])}
                        </ul>
                    </div>
            """
        elif "operations" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-screwdriver-wrench"></i> Operations</div>
                        <ul>
                            {"".join([f"<li>{op}</li>" for op in data_flows[i]["operations"]])}
                        </ul>
                    </div>
            """
        elif "parameters" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-sliders"></i> Tuned Parameters</div>
                        <ul>
                            {"".join([f"<li>{param}</li>" for param in data_flows[i]["parameters"]])}
                        </ul>
                    </div>
            """
        elif "tracked" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-chart-bar"></i> Tracked Metrics</div>
                        <ul>
                            {"".join([f"<li>{metric}</li>" for metric in data_flows[i]["tracked"]])}
                        </ul>
                    </div>
            """
        elif "artifacts" in data_flows[i]:
            interactive_flowchart += f"""
                    <div class="detail-section">
                        <div class="detail-label"><i class="fa-solid fa-box-archive"></i> Saved Artifacts</div>
                        <ul>
                            {"".join([f"<li>{art}</li>" for art in data_flows[i]["artifacts"]])}
                        </ul>
                    </div>
            """

        interactive_flowchart += """
                </div>
            </div>
        </div>
        """

        # Add connector arrow between stages
        if i < len(PIPELINE_STAGES) - 1:
            interactive_flowchart += '<div class="arrow-connector"></div>'

    interactive_flowchart += """
    </div>
    
    <script>
        function toggleStage(stageId) {
            // Get all stages
            const allStages = document.querySelectorAll('.pipeline-stage');
            const clickedStage = document.getElementById(stageId);
            
            // If clicked stage is already active, deactivate it
            if (clickedStage.classList.contains('active')) {
                clickedStage.classList.remove('active');
            } else {
                // Deactivate all other stages
                allStages.forEach(stage => stage.classList.remove('active'));
                // Activate clicked stage
                clickedStage.classList.add('active');
            }
        }
    </script>
    """

    # Display the interactive flowchart
    st.components.v1.html(interactive_flowchart, height=2400, scrolling=True)


def show_pipeline_scripts():
    """Display all pipeline scripts with HTML flowchart explanations"""
    st.markdown(
        '<h2 class="section-header"><i class="fa-solid fa-terminal icon"></i>Pipeline Scripts Overview</h2>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    The project includes **three pipeline execution scripts**, each designed for different use cases. 
    Click on any pipeline below to see detailed workflow and execution steps.
    """)

    # Pipeline comparison table
    st.markdown("### 📊 Quick Comparison")

    comparison_html = """
    <div style="margin: 20px 0;">
        <table style="width: 100%; border-collapse: collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <thead>
                <tr style="background-color: #0f766e; color: white;">
                    <th style="padding: 15px; text-align: left; border: 1px solid #ddd;">Pipeline Script</th>
                    <th style="padding: 15px; text-align: left; border: 1px solid #ddd;">Purpose</th>
                    <th style="padding: 15px; text-align: left; border: 1px solid #ddd;">Execution Style</th>
                    <th style="padding: 15px; text-align: left; border: 1px solid #ddd;">Best For</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background-color: #f8fafc;">
                    <td style="padding: 12px; border: 1px solid #ddd;"><strong>run_pipeline.py</strong></td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Full production pipeline</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Sequential, all stages</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Production deployment</td>
                </tr>
                <tr style="background-color: white;">
                    <td style="padding: 12px; border: 1px solid #ddd;"><strong>quick_pipeline.py</strong></td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Fast prototyping</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Sequential, reduced trials</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Rapid testing & iteration</td>
                </tr>
                <tr style="background-color: #f8fafc;">
                    <td style="padding: 12px; border: 1px solid #ddd;"><strong>modular_pipeline.py</strong></td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Stage-by-stage execution</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">CLI-based, selective stages</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">Development & debugging</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    st.markdown(comparison_html, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive pipeline flowcharts
    pipeline_html = """
    <style>
    .pipeline-container {
        margin: 30px 0;
    }
    
    .pipeline-header {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white;
        padding: 20px 25px;
        border-radius: 10px 10px 0 0;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .pipeline-header:hover {
        background: linear-gradient(135deg, #115e59 0%, #0d9488 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .pipeline-title {
        font-size: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .pipeline-content {
        display: none;
        background: white;
        border: 2px solid #e5e7eb;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 30px;
        animation: slideDown 0.3s ease;
    }
    
    .pipeline-content.active {
        display: block;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .flow-stage {
        background: linear-gradient(to right, #f8fafc, #ffffff);
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #0f766e;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    .flow-stage:hover {
        border-left-width: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }
    
    .stage-number {
        display: inline-block;
        background: #0f766e;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: 700;
        margin-right: 12px;
    }
    
    .stage-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a365d;
        margin-bottom: 8px;
    }
    
    .stage-desc {
        color: #475569;
        margin: 8px 0;
        line-height: 1.6;
    }
    
    .stage-arrow {
        text-align: center;
        color: #0f766e;
        font-size: 1.8rem;
        margin: 10px 0;
    }
    
    .code-snippet {
        background: #1e293b;
        color: #e2e8f0;
        padding: 15px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        margin: 15px 0;
        overflow-x: auto;
    }
    
    .highlight {
        background: #fef3c7;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: 600;
    }
    
    .info-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 5px 5px 5px 0;
    }
    
    .toggle-icon {
        transition: transform 0.3s ease;
    }
    
    .toggle-icon.rotated {
        transform: rotate(180deg);
    }
    </style>
    
    <script>
    function togglePipeline(id) {
        const content = document.getElementById(id);
        const icon = document.getElementById(id + '-icon');
        
        // Close all other pipelines
        document.querySelectorAll('.pipeline-content').forEach(el => {
            if (el.id !== id) {
                el.classList.remove('active');
            }
        });
        
        document.querySelectorAll('.toggle-icon').forEach(el => {
            if (el.id !== id + '-icon') {
                el.classList.remove('rotated');
            }
        });
        
        // Toggle current pipeline
        content.classList.toggle('active');
        icon.classList.toggle('rotated');
    }
    </script>
    
    <!-- Pipeline 1: run_pipeline.py -->
    <div class="pipeline-container">
        <div class="pipeline-header" onclick="togglePipeline('pipeline1')">
            <div class="pipeline-title">
                <i class="fa-solid fa-rocket"></i>
                <span>1. run_pipeline.py - Full Production Pipeline</span>
            </div>
            <i class="fa-solid fa-chevron-down toggle-icon" id="pipeline1-icon"></i>
        </div>
        <div class="pipeline-content" id="pipeline1">
            <h3 style="color: #0f766e; margin-bottom: 20px;">
                <i class="fa-solid fa-info-circle"></i> Overview
            </h3>
            <p style="font-size: 1.05rem; line-height: 1.7; color: #334155;">
                The <strong>full production pipeline</strong> executes all 8 stages sequentially with production-ready parameters.
                It includes comprehensive error handling, state management, and detailed logging for deployment scenarios.
            </p>
            
            <div class="info-badge">677 lines</div>
            <div class="info-badge">100 Optuna trials</div>
            <div class="info-badge">5 training runs</div>
            <div class="info-badge">YAML config support</div>
            
            <div class="code-snippet">
python scripts/run_pipeline.py
            </div>
            
            <h4 style="margin-top: 30px; color: #1a365d;">Pipeline Flow:</h4>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">1</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Initialize Configuration</div>
                        <div class="stage-desc">
                            Load configuration from <span class="highlight">pipeline_config.yaml</span> or use defaults.
                            Set up paths, parameters, and logging.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">2</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Load Data</div>
                        <div class="stage-desc">
                            Read train.csv and test.csv from <span class="highlight">data/raw/</span> directory.
                            Validate file existence and basic structure.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">3</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Validate Data</div>
                        <div class="stage-desc">
                            Run Great Expectations validation suite. Check schema, data types, 
                            ranges, and business logic constraints.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">4</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Preprocess Raw Data</div>
                        <div class="stage-desc">
                            Clean column names, handle missing values using median/mode imputation.
                            Output: cleaned dataframes ready for feature engineering.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">5</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Build Features</div>
                        <div class="stage-desc">
                            Create 10 engineered features: CLV, risk scores, engagement metrics, 
                            support efficiency, and interaction-based features.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">6</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Preprocess Features</div>
                        <div class="stage-desc">
                            Encode categorical variables (Label/OneHot), scale numerical features.
                            Split X and y, prepare final training matrices.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">7</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Optuna Hyperparameter Tuning</div>
                        <div class="stage-desc">
                            Run <span class="highlight">100 trials</span> to optimize XGBoost hyperparameters.
                            Objective: maximize recall score on validation set.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">8</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Train & Track with MLflow</div>
                        <div class="stage-desc">
                            Train <span class="highlight">5 runs</span> with best parameters.
                            Log metrics, parameters, and models to MLflow. Select best model by recall.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">9</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Save Final Model</div>
                        <div class="stage-desc">
                            Save best model to <span class="highlight">models/</span> directory.
                            Include metadata, preprocessing artifacts, and performance metrics.
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="background: #f0fdf4; border: 2px solid #86efac; border-radius: 8px; padding: 20px; margin-top: 30px;">
                <h4 style="color: #166534; margin-bottom: 10px;">
                    <i class="fa-solid fa-check-circle"></i> Use Cases
                </h4>
                <ul style="color: #166534; line-height: 2;">
                    <li>Production model training with full optimization</li>
                    <li>Scheduled/automated ML pipeline execution</li>
                    <li>Final model deployment preparation</li>
                    <li>Comprehensive experiment tracking</li>
                </ul>
            </div>
        </div>
    </div>
    
    <!-- Pipeline 2: quick_pipeline.py -->
    <div class="pipeline-container">
        <div class="pipeline-header" onclick="togglePipeline('pipeline2')">
            <div class="pipeline-title">
                <i class="fa-solid fa-bolt"></i>
                <span>2. quick_pipeline.py - Fast Prototyping Pipeline</span>
            </div>
            <i class="fa-solid fa-chevron-down toggle-icon" id="pipeline2-icon"></i>
        </div>
        <div class="pipeline-content" id="pipeline2">
            <h3 style="color: #0f766e; margin-bottom: 20px;">
                <i class="fa-solid fa-info-circle"></i> Overview
            </h3>
            <p style="font-size: 1.05rem; line-height: 1.7; color: #334155;">
                The <strong>quick pipeline</strong> is a streamlined version optimized for rapid iteration and testing.
                It reduces computation time by using fewer trials and runs while maintaining the full pipeline structure.
            </p>
            
            <div class="info-badge">125 lines</div>
            <div class="info-badge">50 Optuna trials</div>
            <div class="info-badge">3 training runs</div>
            <div class="info-badge">~70% faster</div>
            
            <div class="code-snippet">
python scripts/quick_pipeline.py
            </div>
            
            <h4 style="margin-top: 30px; color: #1a365d;">Simplified Flow:</h4>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">1</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Quick Load & Validate</div>
                        <div class="stage-desc">
                            Fast data loading with basic validation checks only.
                            Skips extensive Great Expectations suite for speed.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">2</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Preprocess & Engineer</div>
                        <div class="stage-desc">
                            Combined preprocessing and feature engineering step.
                            Same transformations as full pipeline, executed sequentially.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">3</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Quick Optuna Tuning</div>
                        <div class="stage-desc">
                            Reduced to <span class="highlight">50 trials</span> (vs 100 in full pipeline).
                            Still finds near-optimal hyperparameters with less time.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">4</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Fast Training</div>
                        <div class="stage-desc">
                            Train <span class="highlight">3 runs</span> (vs 5 in full pipeline) with best parameters.
                            MLflow tracking still enabled for experiment comparison.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stage-arrow">↓</div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">5</span>
                    <div style="flex: 1;">
                        <div class="stage-title">Save & Report</div>
                        <div class="stage-desc">
                            Save best model and generate quick performance summary.
                            Ready for immediate testing and iteration.
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="background: #fef3c7; border: 2px solid #fbbf24; border-radius: 8px; padding: 20px; margin-top: 30px;">
                <h4 style="color: #92400e; margin-bottom: 10px;">
                    <i class="fa-solid fa-lightbulb"></i> Use Cases
                </h4>
                <ul style="color: #92400e; line-height: 2;">
                    <li>Rapid prototyping and experimentation</li>
                    <li>Testing pipeline changes quickly</li>
                    <li>Feature engineering validation</li>
                    <li>Development environment testing</li>
                </ul>
            </div>
        </div>
    </div>
    
    <!-- Pipeline 3: modular_pipeline.py -->
    <div class="pipeline-container">
        <div class="pipeline-header" onclick="togglePipeline('pipeline3')">
            <div class="pipeline-title">
                <i class="fa-solid fa-cubes"></i>
                <span>3. modular_pipeline.py - Stage-by-Stage Execution</span>
            </div>
            <i class="fa-solid fa-chevron-down toggle-icon" id="pipeline3-icon"></i>
        </div>
        <div class="pipeline-content" id="pipeline3">
            <h3 style="color: #0f766e; margin-bottom: 20px;">
                <i class="fa-solid fa-info-circle"></i> Overview
            </h3>
            <p style="font-size: 1.05rem; line-height: 1.7; color: #334155;">
                The <strong>modular pipeline</strong> provides fine-grained control with a CLI interface.
                Execute individual stages, skip stages, or run from a specific point - perfect for debugging and development.
            </p>
            
            <div class="info-badge">435 lines</div>
            <div class="info-badge">CLI interface</div>
            <div class="info-badge">State persistence</div>
            <div class="info-badge">Selective execution</div>
            
            <div class="code-snippet">
# Run single stage<br/>
python scripts/modular_pipeline.py --stages load<br/><br/>
# Run multiple stages<br/>
python scripts/modular_pipeline.py --stages load validate preprocess<br/><br/>
# Run all stages<br/>
python scripts/modular_pipeline.py --all<br/><br/>
# Skip certain stages<br/>
python scripts/modular_pipeline.py --all --skip validate<br/><br/>
# Resume from specific stage<br/>
python scripts/modular_pipeline.py --from features
            </div>
            
            <h4 style="margin-top: 30px; color: #1a365d;">Available Stages:</h4>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">1</span>
                    <div style="flex: 1;">
                        <div class="stage-title">load - Load Data</div>
                        <div class="stage-desc">
                            Loads train.csv and test.csv. Saves state to <span class="highlight">pipeline_state.pkl</span>.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--stages load
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">2</span>
                    <div style="flex: 1;">
                        <div class="stage-title">validate - Validate Data Quality</div>
                        <div class="stage-desc">
                            Runs Great Expectations validation. Can be skipped during development.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--stages validate --skip validate  # Skip this stage
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">3</span>
                    <div style="flex: 1;">
                        <div class="stage-title">preprocess - Raw Preprocessing</div>
                        <div class="stage-desc">
                            Column cleaning and missing value imputation. State preserved between stages.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--stages preprocess
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">4</span>
                    <div style="flex: 1;">
                        <div class="stage-title">features - Feature Engineering</div>
                        <div class="stage-desc">
                            Build 10 engineered features. Test changes to feature logic quickly.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--from features  # Resume from here
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">5</span>
                    <div style="flex: 1;">
                        <div class="stage-title">encode - Feature Preprocessing</div>
                        <div class="stage-desc">
                            Encode and scale features. Prepare final X and y matrices.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--stages encode
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">6</span>
                    <div style="flex: 1;">
                        <div class="stage-title">optuna - Hyperparameter Optimization</div>
                        <div class="stage-desc">
                            Optuna tuning with configurable trials via <span class="highlight">--n-trials</span>.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--stages optuna --n-trials 20  # Quick tuning
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">7</span>
                    <div style="flex: 1;">
                        <div class="stage-title">train - Model Training</div>
                        <div class="stage-desc">
                            Train with MLflow tracking. Control runs via <span class="highlight">--n-runs</span>.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--stages train --n-runs 2  # Quick training
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flow-stage">
                <div style="display: flex; align-items: start;">
                    <span class="stage-number">8</span>
                    <div style="flex: 1;">
                        <div class="stage-title">save - Save Model</div>
                        <div class="stage-desc">
                            Save final model and artifacts. Last stage in the pipeline.
                        </div>
                        <div class="code-snippet" style="margin-top: 10px; font-size: 0.85rem;">
--stages save
                        </div>
                    </div>
                </div>
            </div>
            
            <h4 style="margin-top: 30px; color: #1a365d;">Advanced CLI Features:</h4>
            
            <div style="background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <h5 style="color: #0f766e; margin-bottom: 15px;">State Management</h5>
                <p style="color: #475569; margin-bottom: 10px;">
                    Pipeline state is automatically saved after each stage to <code>pipeline_state.pkl</code>.
                    Resume execution from any point without re-running completed stages.
                </p>
                <div class="code-snippet">
# Run first 3 stages<br/>
python scripts/modular_pipeline.py --stages load validate preprocess<br/><br/>
# Later: continue from features<br/>
python scripts/modular_pipeline.py --from features
                </div>
            </div>
            
            <div style="background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <h5 style="color: #0f766e; margin-bottom: 15px;">Selective Execution</h5>
                <p style="color: #475569; margin-bottom: 10px;">
                    Run specific stages for debugging or testing individual components.
                </p>
                <div class="code-snippet">
# Test only feature engineering<br/>
python scripts/modular_pipeline.py --stages features<br/><br/>
# Run all except validation<br/>
python scripts/modular_pipeline.py --all --skip validate
                </div>
            </div>
            
            <div style="background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <h5 style="color: #0f766e; margin-bottom: 15px;">Parameter Control</h5>
                <p style="color: #475569; margin-bottom: 10px;">
                    Override default parameters for optuna and training stages.
                </p>
                <div class="code-snippet">
# Fast development testing<br/>
python scripts/modular_pipeline.py --all --n-trials 5 --n-runs 2<br/><br/>
# Production with more trials<br/>
python scripts/modular_pipeline.py --all --n-trials 200 --n-runs 10
                </div>
            </div>
            
            <div style="background: #ede9fe; border: 2px solid #a78bfa; border-radius: 8px; padding: 20px; margin-top: 30px;">
                <h4 style="color: #5b21b6; margin-bottom: 10px;">
                    <i class="fa-solid fa-star"></i> Use Cases
                </h4>
                <ul style="color: #5b21b6; line-height: 2;">
                    <li>Debugging specific pipeline stages</li>
                    <li>Testing feature engineering changes</li>
                    <li>CI/CD pipeline testing (used in GitHub Actions)</li>
                    <li>Iterative model development</li>
                    <li>Resuming failed pipeline runs</li>
                </ul>
            </div>
        </div>
    </div>
    
    <!-- Summary Section -->
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; padding: 30px; margin-top: 40px; border: 2px solid #cbd5e1;">
        <h3 style="color: #1a365d; margin-bottom: 20px;">
            <i class="fa-solid fa-graduation-cap"></i> Which Pipeline Should I Use?
        </h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 25px;">
            <div style="background: white; padding: 20px; border-radius: 8px; border: 2px solid #10b981;">
                <div style="font-size: 2rem; text-align: center; margin-bottom: 10px;">🚀</div>
                <h4 style="color: #10b981; text-align: center; margin-bottom: 15px;">Production</h4>
                <p style="text-align: center; color: #475569;">
                    Use <strong>run_pipeline.py</strong> for final model training and deployment
                </p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 8px; border: 2px solid #f59e0b;">
                <div style="font-size: 2rem; text-align: center; margin-bottom: 10px;">⚡</div>
                <h4 style="color: #f59e0b; text-align: center; margin-bottom: 15px;">Development</h4>
                <p style="text-align: center; color: #475569;">
                    Use <strong>quick_pipeline.py</strong> for fast iteration and prototyping
                </p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 8px; border: 2px solid #8b5cf6;">
                <div style="font-size: 2rem; text-align: center; margin-bottom: 10px;">🔧</div>
                <h4 style="color: #8b5cf6; text-align: center; margin-bottom: 15px;">Debugging</h4>
                <p style="text-align: center; color: #475569;">
                    Use <strong>modular_pipeline.py</strong> for stage-by-stage execution
                </p>
            </div>
        </div>
    </div>
    """

    st.components.v1.html(pipeline_html, height=3000, scrolling=True)


def show_module_explorer():
    """Display module explorer with buttons"""
    st.markdown(
        '<h2 class="section-header"><i class="fa-solid fa-folder-tree icon"></i>Module Explorer</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "Explore each module and its functions. Click on a module to see its functions."
    )

    # Module selector
    modules = list(FUNCTIONS_DATA.keys())

    for module_name in modules:
        st.markdown(f"### {module_name}")

        module_data = FUNCTIONS_DATA[module_name]
        files = list(module_data.keys())

        cols = st.columns(len(files))

        for idx, file_name in enumerate(files):
            with cols[idx]:
                if st.button(
                    f"{file_name}",
                    key=f"btn_{module_name}_{file_name}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state["selected_file"] = (module_name, file_name)

        # Show selected file functions
        if "selected_file" in st.session_state:
            sel_module, sel_file = st.session_state["selected_file"]
            if sel_module == module_name:
                st.markdown(f"#### Functions in `{sel_file}`:")

                for func_data in module_data[sel_file]:
                    with st.container():
                        st.markdown(f"##### `{func_data['name']}()`")
                        st.markdown(func_data["description"])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Inputs:**")
                            for inp in func_data["inputs"]:
                                st.code(inp)

                        with col2:
                            st.markdown("**Output:**")
                            for out in func_data["outputs"]:
                                st.code(out)

                        st.markdown("---")

        st.markdown("---")


def show_function_details():
    """Display detailed function information"""
    st.markdown(
        '<h2 class="section-header"><i class="fa-solid fa-file-code icon"></i>Function Details</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "Detailed view of all functions with their inputs, arguments, and outputs."
    )

    # Module selector
    module_name = st.selectbox("Select Module", list(FUNCTIONS_DATA.keys()))

    if module_name:
        module_data = FUNCTIONS_DATA[module_name]
        file_name = st.selectbox("Select File", list(module_data.keys()))

        if file_name:
            st.markdown(f"### Functions in `{module_name}/{file_name}`")

            for func_data in module_data[file_name]:
                st.markdown(f"#### `{func_data['name']}()`")

                # Function description
                st.info(func_data["description"])

                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Inputs", "Arguments", "Returns", "Example"]
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
    print("Data validation passed!")
else:
    print("Validation failed:")
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
