import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from xgboost.core import XGBoostError


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
        if {"YearBuilt", "YearRemodAdd"}.issubset(data.columns):
            data["IsRemodeled"] = (
                data["YearBuilt"] != data["YearRemodAdd"]
            ).astype(int)
        if "PoolArea" in data.columns:
            data["HasPool"] = (data["PoolArea"].fillna(0) > 0).astype(int)
        if "GarageArea" in data.columns:
            data["HasGarage"] = (data["GarageArea"].fillna(0) > 0).astype(int)
        if "TotalBsmtSF" in data.columns:
            data["HasBasement"] = (data["TotalBsmtSF"].fillna(0) > 0).astype(int)
        if "Fireplaces" in data.columns:
            data["HasFireplace"] = (data["Fireplaces"].fillna(0) > 0).astype(int)
        if {"TotRmsAbvGrd", "FullBath", "HalfBath"}.issubset(data.columns):
            data["TotalRooms"] = (
                data["TotRmsAbvGrd"] + data["FullBath"] + data["HalfBath"]
            )
        if "OverallQual" in data.columns:
            data["QualityArea"] = data["OverallQual"] * data["TotalSF"]

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


def build_xgb_model(device: str) -> XGBRegressor:
    return XGBRegressor(
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
        device=device,
        eval_metric="rmse",
    )


def build_model(
    train: pd.DataFrame, test: pd.DataFrame, device: str = "auto"
) -> TransformedTargetRegressor:
    resolved_device = "cuda" if device == "auto" else device
    model = build_xgb_model(resolved_device)
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


def build_ridge_model(
    train: pd.DataFrame, test: pd.DataFrame
) -> TransformedTargetRegressor:
    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train, test)),
            ("model", RidgeCV(alphas=[0.1, 1.0, 3.0, 10.0, 30.0])),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def use_cpu_model(
    estimator: TransformedTargetRegressor,
) -> TransformedTargetRegressor:
    cpu_model = clone(estimator)
    cpu_model.regressor.named_steps["model"].set_params(device="cpu")
    return cpu_model


def split_features_target(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return train.drop(columns=["SalePrice"]), train["SalePrice"]


def filter_training_outliers(train: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"GrLivArea", "SalePrice"}
    if not required_columns.issubset(train.columns):
        return train.copy()

    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    return train.loc[~outlier_mask].copy()


def rmsle(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    clipped = np.maximum(np.asarray(y_pred), 0)
    return float(np.sqrt(mean_squared_log_error(y_true, clipped)))


def blend_predictions(
    xgb_predictions: Iterable[float],
    ridge_predictions: Iterable[float],
    xgb_weight: float,
) -> np.ndarray:
    xgb_values = np.asarray(xgb_predictions, dtype=float)
    ridge_values = np.asarray(ridge_predictions, dtype=float)
    return xgb_weight * xgb_values + (1 - xgb_weight) * ridge_values


def cross_validate_rmsle(
    estimator: TransformedTargetRegressor,
    train: pd.DataFrame,
    folds: int = 5,
    allow_cpu_fallback: bool = True,
) -> List[float]:
    x, y = split_features_target(train)
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_index, validation_index in cv.split(x):
        fold_model = clone(estimator)
        try:
            fold_model.fit(x.iloc[train_index], y.iloc[train_index])
        except XGBoostError:
            if not allow_cpu_fallback:
                raise
            fold_model = use_cpu_model(fold_model)
            fold_model.fit(x.iloc[train_index], y.iloc[train_index])
        predictions = fold_model.predict(x.iloc[validation_index])
        scores.append(rmsle(y.iloc[validation_index], predictions))

    return scores


def cross_validate_blend(
    xgb_estimator: TransformedTargetRegressor,
    ridge_estimator: TransformedTargetRegressor,
    train: pd.DataFrame,
    folds: int = 5,
    weights: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6),
    allow_cpu_fallback: bool = True,
) -> Tuple[float, Dict[float, List[float]], Dict[float, pd.Series]]:
    x, y = split_features_target(train)
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores_by_weight = {weight: [] for weight in weights}
    oof_predictions_by_weight = {
        weight: pd.Series(index=train.index, dtype=float) for weight in weights
    }

    for train_index, validation_index in cv.split(x):
        validation_labels = y.iloc[validation_index]
        validation_index_values = x.iloc[validation_index].index
        fold_xgb = clone(xgb_estimator)
        fold_ridge = clone(ridge_estimator)

        try:
            fold_xgb.fit(x.iloc[train_index], y.iloc[train_index])
        except XGBoostError:
            if not allow_cpu_fallback:
                raise
            fold_xgb = use_cpu_model(fold_xgb)
            fold_xgb.fit(x.iloc[train_index], y.iloc[train_index])

        fold_ridge.fit(x.iloc[train_index], y.iloc[train_index])

        xgb_predictions = fold_xgb.predict(x.iloc[validation_index])
        ridge_predictions = fold_ridge.predict(x.iloc[validation_index])

        for weight in weights:
            blended = blend_predictions(xgb_predictions, ridge_predictions, weight)
            scores_by_weight[weight].append(rmsle(validation_labels, blended))
            oof_predictions_by_weight[weight].loc[validation_index_values] = blended

    best_weight = min(
        scores_by_weight,
        key=lambda weight: float(np.mean(scores_by_weight[weight])),
    )
    return best_weight, scores_by_weight, oof_predictions_by_weight


def build_error_analysis(
    train: pd.DataFrame, predictions: Iterable[float]
) -> pd.DataFrame:
    engineered = FeatureEngineer().fit_transform(train)
    prediction_values = np.maximum(np.asarray(predictions, dtype=float), 0)

    report = pd.DataFrame(
        {
            "Id": train["Id"].astype(int) if "Id" in train.columns else train.index,
            "SalePrice": train["SalePrice"].to_numpy(dtype=float),
            "Prediction": prediction_values,
            "AbsLogError": np.abs(
                np.log1p(train["SalePrice"].to_numpy(dtype=float))
                - np.log1p(prediction_values)
            ),
        },
        index=train.index,
    )
    diagnostic_columns = [
        "GrLivArea",
        "TotalSF",
        "OverallQual",
        "Neighborhood",
        "YearBuilt",
        "GarageArea",
    ]
    for column in diagnostic_columns:
        if column in engineered.columns:
            report[column] = engineered[column]

    return report.sort_values("AbsLogError", ascending=False).reset_index(drop=True)


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


def train_and_predict(
    data_dir: Path,
    output_path: Path,
    device: str = "auto",
    folds: int = 5,
    blend_weights: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6),
    error_analysis_output: Optional[Path] = None,
    remove_outliers: bool = False,
) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    if remove_outliers:
        original_count = len(train)
        train = filter_training_outliers(train)
        print(f"Removed {original_count - len(train)} training outliers")

    xgb_model = build_model(train, test, device=device)
    ridge_model = build_ridge_model(train, test)
    best_weight, scores_by_weight, oof_predictions_by_weight = cross_validate_blend(
        xgb_model,
        ridge_model,
        train,
        folds=folds,
        weights=blend_weights,
    )
    print("Blend CV RMSLE:")
    for weight, scores in scores_by_weight.items():
        print(
            f"xgb_weight={weight:.2f}: "
            f"{np.mean(scores):.5f} +/- {np.std(scores):.5f} "
            f"folds={', '.join(f'{score:.5f}' for score in scores)}"
        )
    print(f"Best xgb_weight={best_weight:.2f}")
    if error_analysis_output is not None:
        error_report = build_error_analysis(
            train,
            oof_predictions_by_weight[best_weight].reindex(train.index).to_numpy(),
        )
        error_report.to_csv(error_analysis_output, index=False)
        print(f"Wrote {error_analysis_output}")

    x, y = split_features_target(train)
    try:
        xgb_model.fit(x, y)
    except XGBoostError:
        xgb_model = use_cpu_model(xgb_model)
        xgb_model.fit(x, y)
    ridge_model.fit(x, y)
    predictions = blend_predictions(
        xgb_model.predict(test),
        ridge_model.predict(test),
        xgb_weight=best_weight,
    )
    save_submission(test, predictions, output_path)
    print(f"Wrote {output_path}")


def parse_blend_weights(value: str) -> List[float]:
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights:
        raise argparse.ArgumentTypeError("At least one blend weight is required.")
    invalid_weights = [weight for weight in weights if weight < 0 or weight > 1]
    if invalid_weights:
        raise argparse.ArgumentTypeError("Blend weights must be between 0 and 1.")
    return weights


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="XGBoost device. auto tries CUDA and falls back to CPU if needed.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--blend-weights",
        type=parse_blend_weights,
        default=[1.0, 0.9, 0.8, 0.7, 0.6],
        help="Comma-separated XGBoost weights to compare for XGB/Ridge blending.",
    )
    parser.add_argument(
        "--error-analysis-output",
        type=Path,
        default=Path("error_analysis.csv"),
        help="Path for the OOF validation error report.",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove classic high-area, low-price Ames outliers before training.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train_and_predict(
        args.data_dir,
        args.output,
        device=args.device,
        folds=args.folds,
        blend_weights=args.blend_weights,
        error_analysis_output=args.error_analysis_output,
        remove_outliers=args.remove_outliers,
    )
