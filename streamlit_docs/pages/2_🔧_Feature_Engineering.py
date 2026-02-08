"""
Page 2: Feature Engineering Documentation
Covers feature creation and preprocessing modules
"""

import streamlit as st

st.set_page_config(page_title="Feature Engineering", page_icon="🔧", layout="wide")

# Custom CSS (same as previous page)
st.markdown(
    """
<style>
    .file-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #F57C00;
        margin-top: 2rem;
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #F57C00;
    }
    .function-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .function-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #0D47A1;
        font-size: 1.2rem;
    }
    .param {
        background-color: #FFF9C4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    .return {
        background-color: #C8E6C9;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("🔧 Feature Engineering")
st.markdown("### Transform clean data into powerful ML features")
st.markdown("---")

# Overview
st.markdown("""
## 📋 Overview

This section covers:
- 🎨 **Feature Creation**: Build 10+ engineered features
- 🔄 **Feature Preprocessing**: Encode, scale, and prepare for modeling

**Location**: `src/features/`

**Impact**: Feature engineering can dramatically improve model performance!
""")

st.markdown("---")

# ============================================================================
# FILE 1: build_feature.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/features/build_feature.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Create engineered features from raw customer data

**Location**: `src/features/build_feature.py`

**Dependencies**: `pandas`, `numpy`

**Key Insight**: Creates 10 new features that capture customer behavior patterns
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 build_feature()</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Main function that creates all engineered features from base columns

**Parameters**:
- <span class="param">df</span>: `pd.DataFrame` - Clean DataFrame with base features

**Returns**: <span class="return">pd.DataFrame</span> - DataFrame with original + engineered features

**Use Case**:
```python
# After preprocessing
feature_df = build_feature(clean_df)
print(f"Original features: {df.shape[1]}")
print(f"After engineering: {feature_df.shape[1]}")
# Typically adds 10+ new columns
```

**Created Features** (each explained below):

### 1️⃣ Customer Lifetime Value (CLV)
```python
df["clv"] = df["total_spend"] * (df["tenure"] / 12)
```
**Logic**: Estimates customer's total value over time
- Multiplies total spend by years as customer (tenure/12)
- Higher CLV = more valuable customer = less likely to churn
- Example: $1,000 spend × 2 years = $2,000 CLV

### 2️⃣ Support Efficiency
```python
df["support_efficiency"] = df["support_calls"] / (df["tenure"] + 1)
```
**Logic**: Support calls per month of tenure
- More calls per unit time = more problems = higher churn risk
- +1 prevents division by zero for new customers
- Example: 12 calls / 6 months = 2 calls/month (high)

### 3️⃣ Payment Reliability
```python
df["payment_reliability"] = 1 / (df["payment_delay"] + 1)
```
**Logic**: Inverse of payment delay (higher = more reliable)
- 0 delay = 1.0 (perfect)
- 30 day delay = 0.032 (poor)
- Captures payment behavior

### 4️⃣ Usage Score
```python
df["usage_score"] = df["usage_frequency"] / df["usage_frequency"].max() * 100
```
**Logic**: Normalized usage on 0-100 scale
- Compares customer to most active user
- 100 = highest usage in dataset
- Low usage = potential churn risk

### 5️⃣ Engagement Index
```python
df["engagement_index"] = (df["usage_frequency"] + df["support_calls"]) / 2
```
**Logic**: Combined interaction metric
- Average of usage and support interactions
- Measures overall engagement with service
- Low engagement = higher churn risk

### 6️⃣ Spend per Interaction
```python
df["spend_per_interaction"] = df["total_spend"] / (df["last_interaction"] + 1)
```
**Logic**: How much they spend relative to recency
- Recent interaction + high spend = engaged customer
- Old interaction + low spend = potential churner
- Captures engagement quality

### 7️⃣ Risk Score (Composite)
```python
df["risk_score"] = (
    (df["payment_delay"] / df["payment_delay"].max()) * 0.3 +
    (1 - df["usage_frequency"] / df["usage_frequency"].max()) * 0.3 +
    (df["tenure"] / df["tenure"].max()) * (-0.2) +
    (df["support_calls"] / df["support_calls"].max()) * 0.2
)
```
**Logic**: Weighted risk score combining multiple signals
- **Payment Delay** (30% weight): Late payments = risk
- **Low Usage** (30% weight): Not using service = risk
- **Short Tenure** (-20% weight): New customers = less risk (negative weight)
- **High Support Calls** (20% weight): Many problems = risk
- Range: 0 (low risk) to 1 (high risk)

### 8️⃣ Tenure Category (Binned)
```python
df["tenure_category"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 36, 48, 60],
    labels=["0-12M", "12-24M", "24-36M", "36-48M", "48M+"]
)
```
**Logic**: Group customers by tenure length
- Captures non-linear relationship with churn
- New customers (0-12M) behave differently than long-term (48M+)
- Creates categorical groups for encoding

### 9️⃣ Age Group (Binned)
```python
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["18-30", "30-40", "40-50", "50-60", "60+"]
)
```
**Logic**: Group customers by age demographics
- Different age groups have different churn patterns
- Young adults vs. seniors behave differently
- Creates categories for analysis

### 🔟 Spend Category (Binned)
```python
df["spend_category"] = pd.cut(
    df["total_spend"],
    bins=3,
    labels=["Low", "Medium", "High"]
)
```
**Logic**: Segment customers by spending level
- Low spenders = higher churn risk
- High spenders = more invested = less likely to churn
- 3 equal-width bins

**Feature Summary**:
- **7 Continuous features**: clv, support_efficiency, payment_reliability, usage_score, engagement_index, spend_per_interaction, risk_score
- **3 Categorical features**: tenure_category, age_group, spend_category (will be one-hot encoded next)

**Impact**: These features capture customer behavior, engagement, and risk - much more powerful than raw features alone!
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE 2: feature_preprocess.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/features/feature_preprocess.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Preprocess features for model training (encoding, scaling)

**Location**: `src/features/feature_preprocess.py`

**Dependencies**: `pandas`, `numpy`, `sklearn.preprocessing`

**Why Important**: Converts features into model-ready numerical format
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 preprocess_features()</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Complete feature preprocessing including encoding and scaling

**Parameters**:
- <span class="param">df</span>: `pd.DataFrame` - DataFrame with engineered features
- <span class="param">output_path</span>: `str | Path` - Where to save processed data
- <span class="param">name</span>: `str` - Output filename

**Returns**: <span class="return">pd.DataFrame</span> - Model-ready DataFrame

**Use Case**:
```python
# After build_feature()
processed_df = preprocess_features(
    df=feature_df,
    output_path='data/processed/',
    name='train_processed.csv'
)
```

**Processing Steps**:

### 1️⃣ Label Encode Gender
```python
le_gender = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
```
**What it does**:
- Male → 0
- Female → 1
- Binary encoding (faster than one-hot for binary features)

### 2️⃣ One-Hot Encode Categorical Features
```python
# Creates dummy variables for each category
subscription_dummies = pd.get_dummies(df["subscription_type"], prefix="sub")
# Result: sub_Basic, sub_Standard, sub_Premium

contract_dummies = pd.get_dummies(df["contract_length"], prefix="contract")
# Result: contract_Monthly, contract_Quarterly, contract_Annual

tenure_dummies = pd.get_dummies(df["tenure_category"], prefix="tenuregroup")
# Result: tenuregroup_0-12M, tenuregroup_12-24M, etc.

age_dummies = pd.get_dummies(df["age_group"], prefix="agegroup")
# Result: agegroup_18-30, agegroup_30-40, etc.

spend_dummies = pd.get_dummies(df["spend_category"], prefix="spendcategory")
# Result: spendcategory_Low, spendcategory_Medium, spendcategory_High
```

**Why One-Hot Encoding?**
- Converts categories to binary columns (0 or 1)
- No ordinal relationship assumed
- Example: "Premium" subscription doesn't mean "3x Basic"
- ML models work with numbers, not strings

**Before One-Hot**:
```
subscription_type
-----------------
Premium
Basic
Standard
```

**After One-Hot**:
```
sub_Basic | sub_Standard | sub_Premium
----------|--------------|------------
    0     |      0       |     1
    1     |      0       |     0
    0     |      1       |     0
```

### 3️⃣ Drop Original Categorical Columns
```python
df = df.drop(columns=[
    "subscription_type",
    "contract_length",
    "tenure_category",
    "age_group",
    "spend_category",
    "customerid"
])
```
**Why**: Keep only encoded versions, remove originals (can't train on strings)

### 4️⃣ Scale Numerical Features
```python
# Identify numeric columns (exclude binary and categorical)
features_to_scale = [
    col for col in df.columns
    if col not in {"churn", "gender"} and
    np.issubdtype(df[col].dtype, np.number)
]

# StandardScaler: (value - mean) / std
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
```

**What StandardScaler Does**:
- Centers data: mean = 0
- Normalizes: std = 1
- Formula: `z = (x - μ) / σ`

**Why Scale?**
- Puts all features on same scale
- Age (0-100) vs Spend ($0-$10,000) → different magnitudes
- XGBoost less sensitive, but still helps
- Critical for distance-based models

**Before Scaling**:
```
age: 35    (range: 18-100)
total_spend: 1250  (range: 0-10000)
tenure: 24  (range: 0-60)
```

**After Scaling**:
```
age: 0.23    (normalized)
total_spend: 0.45  (normalized)
tenure: -0.12  (normalized)
```

### 5️⃣ Save Output
```python
output_path.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path / name, index=False)
```
**Saves**: Processed dataframe ready for model training

**Final Feature Count**: Typically **45+ features** from original 12 columns!
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Summary
st.markdown("""
## 🔄 Feature Engineering Pipeline

```
📊 Clean Data (12 base features)
    ↓
🎨 build_feature() - Create 10 engineered features
    ↓ (22 features total)
🔤 Label Encode Gender (Male→0, Female→1)
    ↓
🔢 One-Hot Encode Categoricals (creates 15+ dummy columns)
    ↓
📏 Scale Numerical Features (mean=0, std=1)
    ↓
💾 Save to processed/ directory
    ↓
🎯 Model-Ready DataFrame (45+ features)
```

## 📊 Complete Example

```python
from src.data_processing.preprocess import raw_preprocess
from src.features.build_feature import build_feature
from src.features.feature_preprocess import preprocess_features

# Step 1: Clean data (from previous section)
clean_df = raw_preprocess(raw_df)
print(f"Clean data: {clean_df.shape}")  # (10000, 12)

# Step 2: Engineer features
feature_df = build_feature(clean_df)
print(f"With engineered features: {feature_df.shape}")  # (10000, 22)

# Step 3: Preprocess for modeling
model_ready_df = preprocess_features(
    df=feature_df,
    output_path='data/processed/',
    name='train_processed.csv'
)
print(f"Model-ready data: {model_ready_df.shape}")  # (10000, 45+)

# Features are now:
# - Fully numerical
# - Scaled and normalized
# - Dummy variables created
# - Ready for XGBoost!
```

## 🎯 Key Takeaways

1. **build_feature()**: Creates 10 smart features that capture customer behavior
   - CLV, engagement, risk scores
   - Categorical binning (tenure, age, spend)
   
2. **preprocess_features()**: Converts everything to numbers
   - Encodes categories → binary columns
   - Scales numeric features → standard range
   - Saves processed data for training

**Feature Engineering Impact**:
- 📈 From 12 → 45+ features
- 🎯 Captures non-linear relationships
- 💪 Dramatically improves model performance

**Next Section**: 🎓 Model Training (where we train XGBoost on these features!)
""")

st.markdown("---")
st.markdown("👉 **Navigate to 'Model Training' in the sidebar to continue**")
