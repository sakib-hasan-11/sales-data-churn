"""
Streamlit Multi-Page Project Documentation App
Main entry point - Home page
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Project Docs",
    page_icon="📚",
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
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .section-box {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-header">📚 Churn Prediction Project Documentation</div>',
    unsafe_allow_html=True,
)
st.markdown("### 🎯 Complete End-to-End MLOps Project Guide")

st.markdown("---")

# Introduction
st.markdown("""
## Welcome! 👋

This is your **complete interactive documentation** for the Customer Churn Prediction MLOps project.

Navigate through different sections using the **sidebar** (👈) to explore every component:
""")

# Navigation guide
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📂 Available Sections
    
    1. **📁 Data Processing**
       - Data loading and validation
       - Preprocessing and cleaning
       
    2. **🔧 Feature Engineering**
       - Feature creation
       - Feature preprocessing
       
    3. **🎓 Model Training**
       - MLflow training
       - Optuna hyperparameter tuning
       - Model evaluation
       
    4. **🔮 Inference Engine**
       - Production inference module
       - Preprocessing pipeline
    """)

with col2:
    st.markdown("""
    ### 🚀 Production Components
    
    5. **🌐 API Deployment**
       - FastAPI application
       - Docker & AWS ECS setup
       
    6. **📊 Project Overview**
       - Complete structure
       - Data flow diagrams
       
    7. **🚀 CI/CD & Deployment**
       - GitHub Actions pipeline
       - 140+ automated tests
       - AWS ECR deployment
    """)

st.markdown("---")

# Quick Stats
st.markdown("## 📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="metric-box">
        <h2>12+</h2>
        <p>Python Modules</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="metric-box">
        <h2>30+</h2>
        <p>Functions</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="metric-box">
        <h2>7</h2>
        <p>Main Sections</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
    <div class="metric-box">
        <h2>140+</h2>
        <p>Automated Tests</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Project Overview
st.markdown("""
## 🎯 Project Overview

### **Objective**
Build a production-ready machine learning system to predict customer churn with:
- Automated data pipeline
- Experiment tracking
- REST API deployment
- Cloud-ready containerization

### **Technology Stack**

| Component | Technology |
|-----------|-----------|
| ML Framework | XGBoost, Scikit-learn |
| Experiment Tracking | MLflow |
| Hyperparameter Optimization | Optuna |
| API Framework | FastAPI |
| Containerization | Docker |
| Deployment | AWS ECS, ECR |
| Validation | Great Expectations |
| CI/CD | GitHub Actions |
| Testing | pytest (140+ tests) |

### **Pipeline Flow**

```
📥 Raw Data
    ↓
🧹 Data Preprocessing (clean, handle missing)
    ↓
🔧 Feature Engineering (create 10+ features)
    ↓
🎯 Model Training (XGBoost + Optuna)
    ↓
📊 Evaluation & Tracking (MLflow)
    ↓
💾 Model Registry
    ↓
🔮 Inference API (FastAPI)
    ↓
🐳 Docker Container
    ↓
🧪 CI/CD Pipeline (140+ Tests)
    ↓
☁️ AWS ECR/ECS Deployment
    ↓
🚀 Production API
```
""")

st.markdown("---")

# Key Features
st.markdown("## ✨ Key Features")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    ### 🔄 Data Pipeline
    - Automated preprocessing
    - Column name standardization
    - Missing value handling
    - Data validation with Great Expectations
    """)

with feat_col2:
    st.markdown("""
    ### 🎯 ML Training
    - XGBoost classifier
    - Optuna hyperparameter tuning
    - MLflow experiment tracking
    - Comprehensive metrics (Recall, Precision, F1, AUC)
    """)

with feat_col3:
    st.markdown("""
    ### 🚀 Production Ready
    - FastAPI REST API
    - Batch prediction support
    - Docker containerization
    - AWS ECS deployment config
    - CI/CD with 140+ tests
    - Automated ECR deployment
    """)

st.markdown("---")

# How to use
st.markdown("""
## 📖 How to Use This Documentation

### 🧭 Navigation
- Use the **sidebar** on the left to navigate between sections
- Each page focuses on a specific component
- Pages are organized in the same order as the ML pipeline

### 📝 What You'll Find
Each documentation page includes:
- **📄 File Purpose**: What the module does
- **🔍 Functions**: Every function with detailed explanations
- **💡 Parameters**: Input arguments and their types
- **📤 Returns**: What the function returns
- **🎯 Use Cases**: Real-world examples

### 🎨 Color Coding
- 🔵 **Blue boxes**: Function definitions
- 🟢 **Green headers**: Section titles  
- 🟠 **Orange headers**: File names
- 📦 **Gray boxes**: Additional information

---

### 🚀 Get Started

**Select a section from the sidebar to begin exploring!** 👈

Start with **"Data Processing"** to follow the natural flow of the project, or jump to any section you're interested in.
""")

st.markdown("---")

# Footer
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem; margin-top: 3rem; border-top: 2px solid #eee;'>
    <p style='font-size: 0.9rem;'>
        📚 Churn Prediction Project Documentation | Built with Streamlit<br>
        🔗 Complete MLOps Pipeline from Data to Deployment
    </p>
</div>
""",
    unsafe_allow_html=True,
)
