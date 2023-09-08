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
    df = df.sample(frac=1).reset_index(drop=True)

    yield df


@pytest.fixture(scope="session")
def iris_train(iris) -> pd.DataFrame:
    iris_train = iris.truncate(after=75)

    yield iris_train


@pytest.fixture(scope="session")
def iris_test(iris, iris_train) -> pd.DataFrame:
    iris_test = iris[~iris.index.isin(iris_train.index)]

    yield iris_test


@pytest.fixture(scope="session")
def iris_features(iris_train) -> FeatureAttributesBase:
    features = infer_feature_attributes(iris_train)
    for _, f_value in features.items():
        if f_value["type"] in ["nominal", "ordinal"]:
            f_value["non_sensitive"] = True

    yield features


@pytest.fixture(scope="session")
def iris_trainee(iris_train, iris_features) -> Trainee:
    t = Trainee(features=iris_features)
    t.train(iris_train)
    t.analyze()

    yield t
