(Project README)

Great Expectations validation
----------------------------

To validate the training CSV with the provided expectation suite:

1. Install dependencies (prefer a virtualenv):

```bash
pip install -r requirements.txt
```

2. Run the validator:

```bash
python scripts/validate_train_ge.py --csv data/raw/train.csv \
	--suite great_expectations/expectations/train_suite.yml
```

The script writes `validation_result.json` with a detailed report.

