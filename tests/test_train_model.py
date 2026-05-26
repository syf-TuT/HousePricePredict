from pathlib import Path

import pandas as pd

from train_model import FeatureEngineer, build_preprocessor, save_submission


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
    transformed = preprocessor.fit_transform(train.drop(columns=["SalePrice"]))

    assert transformed.shape[0] == len(train)
    assert transformed.shape[1] >= 3


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
