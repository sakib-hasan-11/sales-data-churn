"""
Page 1: Data Processing Documentation
Covers all data loading, preprocessing, and validation modules
"""

import streamlit as st

st.set_page_config(page_title="Data Processing", page_icon="📁", layout="wide")

# Custom CSS
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
        font-size: 0.9rem;
    }
    .return {
        background-color: #C8E6C9;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("📁 Data Processing")
st.markdown("### All modules related to data loading, cleaning, and validation")
st.markdown("---")

# Overview
st.markdown("""
## 📋 Overview

This section contains modules responsible for:
- 📥 **Loading data** from CSV files
- 🧹 **Preprocessing**: Cleaning column names, handling missing values
- ✅ **Validation**: Ensuring data quality with Great Expectations

**Location**: `src/data_processing/`
""")

st.markdown("---")

# ============================================================================
# FILE 1: load.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/data_processing/load.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Load raw CSV data files into pandas DataFrames

**Location**: `src/data_processing/load.py`

**Dependencies**: `pandas`
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown('<div class="function-name">🔹 load_data()</div>', unsafe_allow_html=True)
st.markdown(
    """
**Description**: Loads a CSV file from a specified path

**Parameters**:
- <span class="param">file_path</span>: `str` - Path to the CSV file
- <span class="param">**kwargs</span>: Additional arguments to pass to pandas `read_csv()`

**Returns**: <span class="return">pd.DataFrame</span> - Loaded dataset

**Use Case**: 
```python
# Load training data
train_df = load_data('data/raw/train.csv')

# Load with specific encoding
test_df = load_data('data/raw/test.csv', encoding='utf-8')
```

**What it does**:
1. Takes a file path as input
2. Uses pandas `read_csv()` to load the data
3. Returns a DataFrame ready for processing
4. Handles CSV parsing automatically
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE 2: preprocess.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/data_processing/preprocess.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Clean and preprocess raw data (handle missing values, standardize columns)

**Location**: `src/data_processing/preprocess.py`

**Dependencies**: `pandas`, `numpy`, `re` (regex)

**Why Important**: Ensures consistent data quality before feature engineering
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 raw_preprocess()</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Complete preprocessing pipeline that cleans column names and fills missing values

**Parameters**:
- <span class="param">df</span>: `pd.DataFrame` - Raw input DataFrame
- <span class="param">strategy</span>: `str` - Fill strategy ('auto', 'constant', 'median', 'mode')
  - `'auto'`: Smart filling (median for numeric, mode for categorical)
  - `'constant'`: Fill with 0 or 'Unknown'
  - `'median'`: Use median for all numeric columns
  - `'mode'`: Use most frequent value
- <span class="param">custom_fill</span>: `Dict[str, Any]` - Custom fill values for specific columns

**Returns**: <span class="return">pd.DataFrame</span> - Cleaned DataFrame

**Use Case**:
```python
# Auto preprocessing
clean_df = raw_preprocess(raw_df, strategy='auto')

# Custom fill for specific columns
custom = {'age': 30, 'gender': 'Unknown'}
clean_df = raw_preprocess(raw_df, custom_fill=custom)
```

**What it does**:
1. **Cleans column names**:
   - Converts "Customer ID" → "customerid"
   - Removes special characters (/, spaces, etc.)
   - Converts to lowercase
   - Standardizes format using regex

2. **Handles missing values**:
   - **Numeric columns**: Fills with median (or 0)
   - **Categorical columns**: Fills with mode (or 'Unknown')
   - **Datetime columns**: Forward fill, then backward fill
   - **Custom fills**: Applies user-specified values first

3. **Strategy Logic**:
   - **'auto'**: Intelligently chooses best method based on data type
   - **Numeric**: Uses median (more robust than mean)
   - **Objects/Strings**: Uses mode (most common value)
   
**Example Flow**:
```
Input: "Customer ID" with missing values
   ↓ (clean names)
"customerid" with missing values
   ↓ (fill missing)
"customerid" with no missing values
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE 3: data_validator.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/utils/data_validator.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Validate data quality using Great Expectations framework

**Location**: `src/utils/data_validator.py`

**Dependencies**: `great_expectations`, `pandas`

**Why Important**: Catches data quality issues before training (missing values, wrong types, outliers)
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 validate_data()</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Validates DataFrame against predefined expectations

**Parameters**:
- <span class="param">df</span>: `pd.DataFrame` - Data to validate
- <span class="param">expectations</span>: `dict` - Validation rules (optional)

**Returns**: <span class="return">ValidationResult</span> - Pass/fail with details

**Use Case**:
```python
# Validate training data
result = validate_data(train_df)

if result.success:
    print("Data is valid!")
else:
    print("Validation failed:", result.errors)
```

**What it does**:
1. **Column Existence**: Checks if required columns are present
2. **Data Types**: Validates column types (int, float, string)
3. **Value Ranges**: Checks if values are within expected ranges
   - Age: 18-100
   - Tenure: >= 0
   - Total_spend: >= 0
4. **Missing Values**: Ensures critical columns have no nulls
5. **Unique Values**: Validates categorical values are in expected set
6. **Statistical Checks**: Mean, std, min/max within bounds

**Validation Rules Example**:
```python
expectations = {
    'age': {
        'type': 'int',
        'min': 18,
        'max': 100,
        'allow_null': False
    },
    'gender': {
        'type': 'str',
        'values': ['Male', 'Female'],
        'allow_null': False
    }
}
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Summary
st.markdown("""
## 🔄 Data Processing Pipeline Flow

```
📥 Raw CSV File (train.csv, test.csv)
    ↓
📂 load_data() - Load into DataFrame
    ↓
🧹 raw_preprocess() - Clean and handle missing values
    ↓
✅ validate_data() - Ensure data quality
    ↓
📊 Clean DataFrame ready for feature engineering
```

## 📊 Example Usage

```python
from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.utils.data_validator import validate_data

# Step 1: Load data
raw_df = load_data('data/raw/train.csv')
print(f"Loaded {len(raw_df)} rows")

# Step 2: Preprocess
clean_df = raw_preprocess(raw_df, strategy='auto')
print(f"Cleaned columns: {list(clean_df.columns)}")

# Step 3: Validate
result = validate_data(clean_df)
if result.success:
    print("✅ Data is valid and ready!")
else:
    print("❌ Validation failed")
    print(result.errors)

# Now ready for feature engineering!
```

## 🎯 Key Takeaways

1. **load_data()**: Simple CSV loader - gets data into Python
2. **raw_preprocess()**: Core cleaning function - standardizes everything
3. **validate_data()**: Quality gate - ensures data meets requirements

**Next Section**: 🔧 Feature Engineering (where we create new features from this clean data)
""")

st.markdown("---")
st.markdown("👉 **Navigate to 'Feature Engineering' in the sidebar to continue**")
