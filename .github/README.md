# GitHub Actions CI Workflows

This folder contains CI/CD workflows for automated testing and validation.

## Workflows

### 1. `workflow.yml` - Standard CI Tests
**Trigger:** Push/PR to main branch  
**Purpose:** Run pytest unit tests  
**Duration:** ~2-3 minutes

```yaml
Steps:
  - Checkout code
  - Setup Python 3.11
  - Install dependencies
  - Run pytest
```

---

### 2. `modular-pipeline-test.yml` - Modular Pipeline CI ⭐

**Trigger:** Push/PR to main/dev branch + Manual  
**Purpose:** Test entire ML pipeline stage-by-stage (CI testing only)  
**Duration:** ~10-15 minutes  
**Data:** Uses `dummy_data.csv` (split 70/30)  
**Artifacts:** All expire after 1 day (testing only, no production models saved)

#### Pipeline Stages Tested:

<table>
<tr><th>Stage</th><th>Job Name</th><th>Parameters</th><th>Output</th></tr>
<tr><td>0</td><td>Prepare Data</td><td>Split dummy_data.csv</td><td>train.csv, test.csv</td></tr>
<tr><td>1</td><td>Load Data</td><td>-</td><td>load.pkl</td></tr>
<tr><td>2</td><td>Validate Data</td><td>-</td><td>validate.pkl</td></tr>
<tr><td>3</td><td>Preprocess</td><td>strategy=auto</td><td>preprocess.pkl</td></tr>
<tr><td>4</td><td>Build Features</td><td>-</td><td>features.pkl</td></tr>
<tr><td>5</td><td>Encode Features</td><td>-</td><td>encode.pkl + processed/</td></tr>
<tr><td>6</td><td>Optuna</td><td><b>5 trials</b></td><td>optuna.pkl</td></tr>
<tr><td>7</td><td>Train Model</td><td><b>2 runs</b></td><td>train.pkl + mlruns/</td></tr>
<tr><td>8</td><td>Save Model</td><td>-</td><td>models/*.pkl</td></tr>
<tr><td>9</td><td>Summary</td><td>-</td><td>Report</td></tr>
</table>

#### Key Features:

✅ **Isolated Testing:** Each stage runs in a separate job  
✅ **Artifact Passing:** Stage outputs passed between jobs  
✅ **Minimal Parameters:** Fast execution (5 trials, 2 runs)  
✅ **Dummy Data:** Safe, reproducible testing  
✅ **Parallel-Ready:** Can be parallelized in future  
✅ **Fail-Fast:** Stops on first failure  

#### Command Used:

```bash
# Each stage runs:
python scripts/modular_pipeline.py --stages <stage_name> --n-trials 5 --n-runs 2
```

#### Artifacts Created:

All artifacts retained for **1 day only** (CI testing):
- `test-data`: Train/test CSV files
- `stage-*`: Intermediate pipeline states
- `processed-data`: Processed feature data
- `mlflow-runs`: MLflow experiment data

**Note:** This is a CI test workflow. No production models are saved. All artifacts automatically deleted after 1 day.

---

## How to Use

### View Workflow Runs:

1. Go to **Actions** tab in GitHub
2. Select workflow from left sidebar
3. Click on any run to see details
4. Download artifacts if needed

### Manual Trigger:

**Modular Pipeline Test** can be triggered manually:

1. Go to **Actions** tab
2. Select "Modular Pipeline CI Test"
3. Click **Run workflow**
4. Choose branch and click **Run**

### Check Results:

```bash
# After pushing to GitHub:
git push origin main

# Then visit:
https://github.com/YOUR_USERNAME/sales-data-churn/actions
```

---

## Data Preparation

### Script: `scripts/prepare_ci_data.py`

Splits `dummy_data.csv` into train/test:

```python
# 70% train, 30% test
# Stratified by Churn column
# Random state: 42 (reproducible)
```

**Usage:**
```bash
python scripts/prepare_ci_data.py
```

**Output:**
- `data/raw/train.csv` (~35 samples)
- `data/raw/test.csv` (~15 samples)

---

## Troubleshooting

### Issue: Workflow fails at specific stage

**View logs:**
1. Click on failed job
2. Expand step to see error
3. Check artifact upload/download

### Issue: "Artifact not found"

**Solution:** Ensure previous stage completed:
- Check job dependencies (`needs:`)
- Verify artifact upload succeeded
- Check retention period (1 day)

### Issue: Test timeout

**Solution:** Parameters are already minimal:
- Optuna: 5 trials (vs 100 production)
- MLflow: 2 runs (vs 5 production)
- Consider skipping validation: `--skip validate`

### Issue: Memory error

**Solution:** Dummy data is small (52 rows), but if issues:
- Reduce trials to 3: `--n-trials 3`
- Reduce runs to 1: `--n-runs 1`
- Skip optuna: `--skip optuna`

---

## Workflow YAML Structure

```yaml
name: Modular Pipeline CI Test

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main, dev ]
  workflow_dispatch:  # Manual trigger

jobs:
  prepare-data:
    # Prepare train/test split
    
  test-stage-load:
    needs: prepare-data
    # Test data loading
    
  test-stage-validate:
    needs: test-stage-load
    # Test validation
    
  # ... more stages ...
  
  test-stage-save:
    needs: test-stage-train
    # Test model saving
    
  pipeline-summary:
    needs: test-stage-save
    if: always()
    # Summary report
```

---

## Performance

**Expected Times (Ubuntu runner):**

| Stage | Duration |
|-------|----------|
| Prepare Data | 30s |
| Load | 45s |
| Validate | 1m |
| Preprocess | 45s |
| Features | 45s |
| Encode | 1m |
| Optuna (5 trials) | 2-3m |
| Train (2 runs) | 2-3m |
| Save | 30s |
| **Total** | **~10-15m** |

---

## Future Enhancements

- [ ] Parallel stage execution where possible
- [ ] Cache dependencies for faster runs
- [ ] Add performance benchmarking
- [ ] Send notifications on failure
- [ ] Generate test coverage report
- [ ] Add stage timing metrics
- [ ] Deploy model to staging on success

---

## Notes

- **Dummy Data:** 52 samples with realistic churn data
- **Stratification:** Maintains class balance in train/test
- **Reproducibility:** Fixed random seed (42)
- **Fast Execution:** Minimal parameters for CI speed
- **Complete Coverage:** Tests every pipeline stage
- **Artifact Passing:** Simulates real pipeline data flow
- **CI Testing Only:** All artifacts expire after 1 day - no production models saved
- **For Production:** Use `run_pipeline.py` or `quick_pipeline.py` locally

---

## Questions?

Check the main project README or review the workflow logs in GitHub Actions.
