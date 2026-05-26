import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(
        self, x: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FeatureEngineer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()

        data["TotalSF"] = self._sum_columns(
            data, ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
        )
        data["TotalBath"] = (
            self._sum_columns(data, ["FullBath", "BsmtFullBath"])
            + 0.5 * self._sum_columns(data, ["HalfBath", "BsmtHalfBath"])
        )
        data["TotalPorchSF"] = self._sum_columns(
            data, ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
        )

        if {"YrSold", "YearBuilt"}.issubset(data.columns):
            data["HouseAge"] = (data["YrSold"] - data["YearBuilt"]).clip(lower=0)
        if {"YrSold", "YearRemodAdd"}.issubset(data.columns):
            data["RemodAge"] = (data["YrSold"] - data["YearRemodAdd"]).clip(lower=0)
        if {"OverallQual", "OverallCond"}.issubset(data.columns):
            data["OverallScore"] = data["OverallQual"] * data["OverallCond"]

        for column in ["MSSubClass", "MoSold"]:
            if column in data.columns:
                data[column] = data[column].astype("string")

        return data

    @staticmethod
    def _sum_columns(data: pd.DataFrame, columns: List[str]) -> pd.Series:
        values = [
            pd.to_numeric(data[column], errors="coerce").fillna(0)
            for column in columns
            if column in data.columns
        ]
        if not values:
            return pd.Series(0, index=data.index, dtype=float)
        return sum(values)


def feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[List[str], List[str]]:
    combined = pd.concat(
        [
            train.drop(columns=["SalePrice"], errors="ignore"),
            test.drop(columns=["SalePrice"], errors="ignore"),
        ],
        axis=0,
        ignore_index=True,
    )
    combined = combined.drop(columns=["Id"], errors="ignore")

    numeric_features = combined.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in combined.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


def build_preprocessor(train: pd.DataFrame, test: pd.DataFrame) -> Pipeline:
    feature_engineer = FeatureEngineer()
    engineered_train = feature_engineer.fit_transform(train)
    engineered_test = feature_engineer.transform(test)
    numeric_features, categorical_features = feature_columns(
        engineered_train, engineered_test
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("columns", column_transformer),
        ]
    )


def build_model(train: pd.DataFrame, test: pd.DataFrame) -> TransformedTargetRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1800,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=1.5,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_alpha=0.001,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device="cuda",
        eval_metric="rmse",
    )
    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train, test)),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def split_features_target(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return train.drop(columns=["SalePrice"]), train["SalePrice"]


def rmsle(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    clipped = np.maximum(np.asarray(y_pred), 0)
    return float(np.sqrt(mean_squared_log_error(y_true, clipped)))


def cross_validate_rmsle(
    estimator: TransformedTargetRegressor,
    train: pd.DataFrame,
    folds: int = 5,
) -> List[float]:
    x, y = split_features_target(train)
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_index, validation_index in cv.split(x):
        fold_model = clone(estimator)
        fold_model.fit(x.iloc[train_index], y.iloc[train_index])
        predictions = fold_model.predict(x.iloc[validation_index])
        scores.append(rmsle(y.iloc[validation_index], predictions))

    return scores


def save_submission(
    test: pd.DataFrame,
    predictions: Iterable[float],
    output_path: Path,
) -> None:
    output = pd.DataFrame(
        {
            "Id": test["Id"].astype(int),
            "SalePrice": np.maximum(np.asarray(predictions, dtype=float), 0),
        }
    )
    output.to_csv(output_path, index=False)


def train_and_predict(data_dir: Path, output_path: Path) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    model = build_model(train, test)
    scores = cross_validate_rmsle(model, train)
    print(
        "CV RMSLE: "
        f"{np.mean(scores):.5f} +/- {np.std(scores):.5f} "
        f"folds={', '.join(f'{score:.5f}' for score in scores)}"
    )

    x, y = split_features_target(train)
    model.fit(x, y)
    predictions = model.predict(test)
    save_submission(test, predictions, output_path)
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Kaggle House Prices model and create submission.csv."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("house-prices-advanced-regression-techniques"),
        help="Directory containing train.csv and test.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Path for the Kaggle submission CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_predict(args.data_dir, args.output)
