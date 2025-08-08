from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from howso.engine import Trainee
from howso.utilities import infer_feature_attributes
from howso.utilities.feature_attributes.infer_feature_attributes import FeatureAttributesBase


@pytest.fixture(scope="session")
def iris() -> pd.DataFrame:
    iris_path = Path(__file__).resolve()
    iris_path = iris_path.parent.parent / "data"
    iris_path = iris_path.resolve()

    df = pd.read_csv(iris_path / "iris.csv")
    return df.sample(frac=1).reset_index(drop=True)


@pytest.fixture(scope="session")
def iris_train(iris) -> pd.DataFrame:
    return iris.truncate(after=75)


@pytest.fixture(scope="session")
def iris_test(iris, iris_train) -> pd.DataFrame:
    return iris[~iris.index.isin(iris_train.index)]


@pytest.fixture(scope="session")
def iris_features(iris_train) -> FeatureAttributesBase:
    features = infer_feature_attributes(iris_train)
    for f_value in features.values():
        if f_value["type"] in ["nominal", "ordinal"]:
            f_value["non_sensitive"] = True

    return features


@pytest.fixture(scope="session")
def iris_trainee(iris_train, iris_features) -> Trainee:
    t = Trainee(features=iris_features)
    t.train(iris_train)
    t.analyze()

    return t
