# Repository Guidelines

## Project Structure & Module Organization

This repository contains a compact Kaggle House Prices workflow. The main implementation is in `train_model.py`, including feature engineering, preprocessing, model construction, cross-validation, blending, error analysis, and submission writing. Tests live in `tests/test_train_model.py`. Competition input files are stored in `house-prices-advanced-regression-techniques/` (`train.csv`, `test.csv`, `sample_submission.csv`, and `data_description.txt`). Generated artifacts such as `submission.csv`, `error_analysis.csv`, and `error_analysis_outliers_removed.csv` are outputs, not source modules.

## Build, Test, and Development Commands

Create and activate a virtual environment before installing dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the full test suite:

```powershell
pytest
```

Train locally and create a Kaggle submission:

```powershell
python train_model.py --device cpu --output submission.csv
```

Use `--device auto` to try CUDA first, `--folds 5` to control cross-validation, `--blend-weights 1.0,0.9,0.8` to compare XGBoost/Ridge blends, and `--remove-outliers` to exclude the classic high-area, low-price outliers.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints where they clarify interfaces, and small pure functions for reusable transformations. Follow the existing naming pattern: classes in `PascalCase` (`FeatureEngineer`), functions and variables in `snake_case` (`build_error_analysis`), and constants in `UPPER_SNAKE_CASE` (`RANDOM_STATE`). Keep pandas transformations explicit and avoid hidden mutation unless the caller clearly owns the data.

## Testing Guidelines

Tests use `pytest` and should be named `test_<behavior>()`. Place new tests in `tests/test_train_model.py` unless the project grows enough to justify splitting by module. Cover feature engineering, argument parsing, output format, and model construction changes. Prefer small in-memory `pandas.DataFrame` fixtures over reading the full Kaggle dataset in unit tests.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries that name the changed behavior, for example `filter_training_outliers` or `add FeatureEngineer`. Keep commits focused on one behavior change or experiment. Pull requests should describe the modeling or data-processing change, list commands run (`pytest`, training command if relevant), and mention any changes to generated outputs or Kaggle submission files.

## Data & Artifact Notes

Do not commit virtual environments, caches, or `__pycache__` directories. Treat the Kaggle input directory as read-only during experiments. When regenerating CSV outputs, include the command and key validation score in the PR so results are reproducible.
