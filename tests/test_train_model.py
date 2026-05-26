from pathlib import Path

import numpy as np
import pandas as pd

from train_model import (
    FeatureEngineer,
    OutOfFoldTargetEncoder,
    blend_predictions,
    build_error_analysis,
    build_model,
    build_preprocessor,
    build_ridge_model,
    build_xgb_model,
    filter_training_outliers,
    parse_args,
    save_submission,
)


def test_feature_engineer_adds_house_price_domain_features():
    data = pd.DataFrame(
        {
            "Id": [1],
            "TotalBsmtSF": [856],
            "1stFlrSF": [856],
            "2ndFlrSF": [854],
            "FullBath": [2],
            "HalfBath": [1],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [0],
            "OpenPorchSF": [61],
            "EnclosedPorch": [0],
            "3SsnPorch": [0],
            "ScreenPorch": [0],
            "YearBuilt": [2003],
            "YearRemodAdd": [2003],
            "YrSold": [2008],
            "OverallQual": [7],
            "OverallCond": [5],
            "PoolArea": [0],
            "GarageArea": [548],
            "Fireplaces": [1],
            "TotRmsAbvGrd": [8],
            "MSSubClass": [60],
        }
    )

    transformed = FeatureEngineer().fit_transform(data)

    assert transformed.loc[0, "TotalSF"] == 2566
    assert transformed.loc[0, "TotalBath"] == 3.5
    assert transformed.loc[0, "TotalPorchSF"] == 61
    assert transformed.loc[0, "HouseAge"] == 5
    assert transformed.loc[0, "RemodAge"] == 5
    assert transformed.loc[0, "OverallScore"] == 35
    assert transformed.loc[0, "IsRemodeled"] == 0
    assert transformed.loc[0, "HasPool"] == 0
    assert transformed.loc[0, "HasGarage"] == 1
    assert transformed.loc[0, "HasBasement"] == 1
    assert transformed.loc[0, "HasFireplace"] == 1
    assert transformed.loc[0, "TotalRooms"] == 11
    assert transformed.loc[0, "QualityArea"] == 17962
    assert transformed.loc[0, "MSSubClass"] == "60"


def test_preprocessor_handles_missing_numeric_and_categorical_values():
    train = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "LotArea": [8450, None, 9600],
            "MSZoning": ["RL", None, "RM"],
            "SalePrice": [208500, 181500, 223500],
        }
    )
    test = pd.DataFrame(
        {
            "Id": [4],
            "LotArea": [11250],
            "MSZoning": ["FV"],
        }
    )

    preprocessor = build_preprocessor(train, test)
    transformed = preprocessor.fit_transform(
        train.drop(columns=["SalePrice"]),
        train["SalePrice"],
    )

    assert transformed.shape[0] == len(train)
    assert transformed.shape[1] >= 3


def test_out_of_fold_target_encoder_uses_fold_only_category_means():
    data = pd.DataFrame({"Neighborhood": ["A", "A", "A", "B", "B", "B"]})
    target = pd.Series([10.0, 20.0, 30.0, 100.0, 110.0, 120.0])

    encoder = OutOfFoldTargetEncoder(
        internal_folds=3,
        smoothing=0.0,
        shuffle=False,
    )
    transformed = encoder.fit_transform(data, target)

    assert transformed["Neighborhood_target_mean"].tolist() == [
        30.0,
        30.0,
        15.0,
        115.0,
        100.0,
        100.0,
    ]


def test_out_of_fold_target_encoder_uses_global_mean_for_unknown_categories():
    train = pd.DataFrame({"Neighborhood": ["A", "A", "B"]})
    target = pd.Series([10.0, 20.0, 100.0])
    validation = pd.DataFrame({"Neighborhood": ["A", "C", None]})

    encoder = OutOfFoldTargetEncoder(internal_folds=2, smoothing=0.0).fit(
        train, target
    )
    transformed = encoder.transform(validation)

    assert transformed["Neighborhood_target_mean"].tolist() == [
        15.0,
        np.mean(target),
        np.mean(target),
    ]


def test_build_xgb_model_accepts_cpu_device():
    model = build_xgb_model("cpu")

    assert model.get_params()["device"] == "cpu"
    assert model.get_params()["tree_method"] == "hist"


def test_build_model_uses_requested_xgboost_device():
    train = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "LotArea": [8450, 9600, 11250, 9550],
            "MSZoning": ["RL", "RL", "RM", "FV"],
            "SalePrice": [208500, 181500, 223500, 140000],
        }
    )
    test = pd.DataFrame(
        {
            "Id": [5],
            "LotArea": [10000],
            "MSZoning": ["RL"],
        }
    )

    model = build_model(train, test, device="cpu")
    xgb_model = model.regressor.named_steps["model"]

    assert xgb_model.get_params()["device"] == "cpu"
    assert xgb_model.get_params()["tree_method"] == "hist"


def test_build_ridge_model_uses_ridgecv():
    train = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "LotArea": [8450, 9600, 11250, 9550],
            "MSZoning": ["RL", "RL", "RM", "FV"],
            "SalePrice": [208500, 181500, 223500, 140000],
        }
    )
    test = pd.DataFrame(
        {
            "Id": [5],
            "LotArea": [10000],
            "MSZoning": ["RL"],
        }
    )

    model = build_ridge_model(train, test)

    assert model.regressor.named_steps["model"].__class__.__name__ == "RidgeCV"


def test_blend_predictions_uses_xgb_weight():
    blended = blend_predictions([100, 200], [80, 220], xgb_weight=0.75)

    assert blended.tolist() == [95.0, 205.0]


def test_parse_args_supports_device_and_folds():
    args = parse_args(["--device", "cpu", "--folds", "3"])

    assert args.device == "cpu"
    assert args.folds == 3


def test_parse_args_supports_blend_weights():
    args = parse_args(["--blend-weights", "1.0,0.8,0.6"])

    assert args.blend_weights == [1.0, 0.8, 0.6]


def test_filter_training_outliers_removes_large_low_price_homes():
    train = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "GrLivArea": [1200, 4500, 4600],
            "SalePrice": [140000, 250000, 650000],
        }
    )

    filtered = filter_training_outliers(train)

    assert filtered["Id"].tolist() == [1, 3]


def test_build_error_analysis_sorts_by_absolute_log_error():
    train = pd.DataFrame(
        {
            "Id": [1, 2],
            "SalePrice": [100000, 200000],
            "GrLivArea": [1200, 2000],
            "TotalBsmtSF": [600, 900],
            "1stFlrSF": [600, 900],
            "2ndFlrSF": [0, 200],
            "OverallQual": [5, 8],
            "Neighborhood": ["NAmes", "NridgHt"],
        }
    )

    report = build_error_analysis(train, [100000, 100000])

    assert report.loc[0, "Id"] == 2
    assert report.loc[0, "Prediction"] == 100000
    assert report.loc[0, "AbsLogError"] > report.loc[1, "AbsLogError"]
    assert "TotalSF" in report.columns


def test_parse_args_supports_error_analysis_and_outlier_filter():
    args = parse_args(
        ["--error-analysis-output", "errors.csv", "--remove-outliers"]
    )

    assert args.error_analysis_output == Path("errors.csv")
    assert args.remove_outliers is True


def test_save_submission_writes_expected_kaggle_format(tmp_path: Path):
    test = pd.DataFrame({"Id": [1461, 1462]})
    output_path = tmp_path / "submission.csv"

    save_submission(test, [125000.25, 175000.75], output_path)

    result = pd.read_csv(output_path)
    assert list(result.columns) == ["Id", "SalePrice"]
    assert result.to_dict("list") == {
        "Id": [1461, 1462],
        "SalePrice": [125000.25, 175000.75],
    }
