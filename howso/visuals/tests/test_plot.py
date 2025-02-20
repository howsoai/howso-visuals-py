import numpy as np
import pandas as pd
import pytest

from howso.visuals import (
    plot_anomalies,
    plot_dataset,
    plot_fairness_disparity,
    plot_interpretable_prediction,
    plot_kl_divergence,
    plot_umap,
)
from howso.visuals.visuals import (
    plot_drift,
    plot_feature_importances,
)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("action_feature", ["sepal-length", "sepal-width"])
@pytest.mark.parametrize("do_generative_reacts", [True, False])
@pytest.mark.parametrize("do_actual_value", [True, False])
@pytest.mark.parametrize("do_residual", [True, False])
def test_plot_interpretable_prediction_react(
    iris_trainee,
    iris_test,
    iris_features,
    action_feature,
    do_generative_reacts,
    do_residual,
    do_actual_value,
):
    num_expected_traces = 3
    context_features = iris_features.get_names(without=[action_feature, "target"])

    predict_case = iris_test.loc[140:140]
    react = iris_trainee.react(
        action_features=[action_feature],
        contexts=predict_case[context_features],
        details={
            "influential_cases": True
        },
    )

    if do_actual_value:
        actual_value = predict_case[action_feature].iloc[0]
        num_expected_traces += 1
    else:
        actual_value = None

    if do_generative_reacts:
        result = iris_trainee.react(
            action_features=[action_feature],
            contexts=predict_case[context_features],
            desired_conviction=5_000,
            num_cases_to_generate=20,
        )
        generative_reacts = result["action"].loc[:, action_feature].values.tolist()
        generative_reacts[0] = generative_reacts[0] + 0.2
        generative_reacts[-1] = generative_reacts[-1] - 0.2
    else:
        generative_reacts = None

    if do_residual:
        residual = iris_trainee.get_prediction_stats(details={"prediction_stats": True, "selected_prediction_stats": ["mae"]})
        residual = residual[action_feature].iloc[0]
    else:
        residual = None

    fig = plot_interpretable_prediction(
        react,
        actual_value=actual_value,
        generative_reacts=generative_reacts,
        residual=residual,
    )

    assert len(list(fig.select_traces())) == num_expected_traces

    if do_residual:
        assert fig.data[1].error_x is not None


@pytest.mark.parametrize("do_most_similar_cases", [True, False])
@pytest.mark.parametrize("do_boundary_cases", [True, False])
@pytest.mark.parametrize("x", ["sepal-width", "petal-width"])
@pytest.mark.parametrize("y", ["sepal-length", "petal-length"])
@pytest.mark.parametrize("highlight_index", [140, [140, 141], None])
@pytest.mark.parametrize("hue", ["target", None])
@pytest.mark.parametrize("title", ["title", None])
def test_plot_dataset(
    iris_trainee,
    iris_test,
    iris_features,
    do_most_similar_cases,
    do_boundary_cases,
    x,
    y,
    highlight_index,
    hue,
    title,
):
    num_expected_traces = 0
    # Grab the highlight case from the test DataFrame. This will add one trace.
    highlight_cases = None
    if highlight_index is not None:
        if not isinstance(highlight_index, list):
            highlight_cases = iris_test.loc[[highlight_index], :]
        else:
            highlight_cases = iris_test.loc[highlight_index, :]

        num_expected_traces += 1

    # We will only get the most similar cases if there"s one or more highlight cases to
    # react to. This will add another trace.
    most_similar_cases = None
    if do_most_similar_cases and highlight_cases is not None:
        most_similar_cases = iris_trainee.react(contexts=highlight_cases, details={"most_similar_cases": True})
        most_similar_cases = pd.concat(
            [pd.DataFrame(mscs) for mscs in most_similar_cases["details"]["most_similar_cases"]]
        ).reset_index(drop=True)
        num_expected_traces += 1

    # Same for bounary cases.
    boundary_cases = None
    if do_boundary_cases and highlight_cases is not None:
        context_features = iris_features.get_names(without=["target"])
        boundary_cases = iris_trainee.react(
            contexts=highlight_cases[context_features],
            action_features=["target"],
            details={"boundary_cases": True}
        )
        boundary_cases = pd.concat(
            [pd.DataFrame(bcs) for bcs in boundary_cases["details"]["boundary_cases"]]
        ).reset_index(drop=True)
        num_expected_traces += 1

    # Hue groups by the requested feature and then creates one trace per group.
    if hue is not None:
        num_expected_traces += iris_test[hue].nunique()
    else:
        num_expected_traces += 1

    fig = plot_dataset(
        iris_test, x, y,
        boundary_cases=boundary_cases,
        most_similar_cases=most_similar_cases,
        highlight_index=highlight_index,
        hue=hue,
        title=title,
    )

    assert len(list(fig.select_traces())) == num_expected_traces


@pytest.fixture(scope="module")
def outliers_convictions(iris_trainee, iris_features):
    iris_trainee.react_into_features(familiarity_conviction_addition=True, distance_contribution=True)
    outliers = iris_trainee.get_cases(
        session=iris_trainee.active_session,
        features=iris_features.get_names() + [
            "familiarity_conviction_addition", ".session_training_index", ".session", "distance_contribution"
        ]
    )

    outliers_indices = outliers[['.session', '.session_training_index']].values
    convictions = iris_trainee.react(
        case_indices=outliers_indices,
        preserve_feature_values=iris_features.get_names(),
        leave_case_out=True,
        details={
            "boundary_cases": True,
            "influential_cases": True,
            "feature_full_residual_convictions_for_case": True,
        }
    )
    convictions = pd.DataFrame(
        convictions["details"]["feature_full_residual_convictions_for_case"]
    )

    yield outliers, convictions


@pytest.mark.parametrize("num_cases_to_plot", [1, 5, 10])
def test_plot_anomalies(
    outliers_convictions,
    num_cases_to_plot
):
    outliers, convictions = outliers_convictions
    fig = plot_anomalies(outliers, convictions, num_cases_to_plot=num_cases_to_plot)
    assert len(fig.data[0].y) == num_cases_to_plot


def test_plot_kl_divergence():
    data = pd.Series([0.5, 1.0, np.inf], index=["a", "b", "c"])
    fig = plot_kl_divergence(data)

    assert fig is not None


@pytest.mark.parametrize("compute_rolling_mean", [True, False])
@pytest.mark.parametrize("line_positions", [[30], [20, 5], None])
@pytest.mark.parametrize("rolling_window", [5, 10, 15])
def test_plot_drift(compute_rolling_mean, line_positions, rolling_window):
    data = pd.DataFrame({"DP": np.arange(5, 15, step=0.5), "GBRF": np.arange(0, 10, step=0.5)})
    fig = plot_drift(
        data,
        compute_rolling_mean=compute_rolling_mean,
        line_positions=line_positions,
        rolling_window=rolling_window,
    )

    assert len(list(fig.select_traces())) == 2


@pytest.mark.parametrize("feature_residuals", [None, 0.5])
def test_plot_feature_importances(feature_residuals):
    data = pd.DataFrame({"a": [0.5], "b": [0.2], "c": [0.1]})
    fig = plot_feature_importances(data, feature_residuals=feature_residuals)

    assert fig is not None


@pytest.mark.parametrize("x_tickangle", [True, False, 45])
def test_plot_fairness_disparity(x_tickangle):
    fairness_results = {
        'Dataset1': {'Male': 0.8, 'Female': 0.7, 'Other': 0.9},
        'Dataset2': {'Male': 0.6, 'Female': 0.9, 'Other': 0.5},
        'Dataset3': {'Male': 0.7, 'Female': 0.4, 'Other': 0.6}
    }
    fig = plot_fairness_disparity(fairness_results, reference_class='Male', x_tickangle=x_tickangle)

    assert fig is not None


@pytest.mark.parametrize("n_cases", [None, 50])
@pytest.mark.parametrize("data", ["iris_train", "iris_trainee"])
def test_plot_umap(data, n_cases, request):
    data = request.getfixturevalue(data)
    fig = plot_umap(data, n_cases=n_cases)

    assert fig is not None
    if n_cases is not None:
        assert len(fig.data[0].y) == n_cases
