from __future__ import annotations

import typing as t
import warnings

import numpy as np
import numpy.typing as npt
from pandas import (
    DataFrame,
    Series,
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import umap

from howso.engine import Trainee
from howso.utilities import infer_feature_attributes

if t.TYPE_CHECKING:
    from howso.engine.trainee import Reaction


def plot_feature_importances(
    feature_importances: DataFrame,
    *,
    num_features_to_plot: t.Optional[int] = None,
    sort_values: bool = True,
    feature_residuals: t.Optional[t.SupportsFloat] = None,
    feature_residuals_zero_bounds: bool = True,
    title: t.Optional[str] = "Feature Importances",
    xaxis_title: t.Optional[str] = "Feature",
    yaxis_title: t.Optional[str] = "Importance",
) -> go.Figure:
    """
    Plot feature importances (either accuracy contributions or prediction contributions).

    Parameters
    ----------
    feature_importances : DataFrame
        A DataFrame containing the feature importance information.
    num_features_to_plot : int, optional
        The number of features to plot importances for. If this is None, all features will be plotted.
    sort_values : bool, default True
        Whether to sort the values before plotting them.
    feature_residuals : SupportsFloat, optional
        The robust feature deviations for calculating and plotting the feature importance error bars.
    feature_residuals_zero_bounds : bool, default True
        If plotting with `feature_residuals`, whether to put a lower bound at 0. Generally set to True
        when plotting prediction contributions and False when plotting accuracy contributions.
    title : str, default "Feature Importances"
        The title for the figure.
    xaxis_title : str, default "Feature"
        The title for the figure's x axis.
    yaxis_title : str, default "Importance"
        The title for the figure's y axis.

    Returns
    -------
    Figure
        The resultant `Plotly` figure.
    """
    if sort_values:
        feature_importances = feature_importances.sort_values(
            by=feature_importances.index[0],
            axis=1,
            ascending=False,
        ).iloc[0]

    if feature_residuals is not None:
        error_value = feature_residuals / len(feature_importances)
        if feature_residuals_zero_bounds:
            # Upper bound = value + error_value
            # Lower bound does not drop below zero
            error_y = {
                "type": "data",
                "array": [error_value] * len(feature_importances),
                # Ensure error doesn't go below 0 for each bar
                "arrayminus": [min(val, error_value) for val in feature_importances],
                "visible": True
            }
        else:
            error_y = {
                "type": "data",
                "array": [error_value] * len(feature_importances),
                "arrayminus": [error_value] * len(feature_importances),
                "visible": True
            }
    else:
        error_y = None

    if num_features_to_plot:
        feature_importances = feature_importances[:num_features_to_plot]
        if error_y is not None:
            error_y["array"] = error_y["array"][:num_features_to_plot]
            if "arrayminus" in error_y:
                error_y["arrayminus"] = error_y["arrayminus"][:num_features_to_plot]

    fig = make_subplots()
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=xaxis_title, tickangle=45),
        yaxis=dict(title=yaxis_title)
    )
    fig.add_trace(
        go.Bar(
            x=feature_importances.index,
            y=feature_importances,
            error_y=error_y
        )
    )

    return fig


def plot_anomalies(
    anomalies: DataFrame,
    convictions: DataFrame,
    *,
    colorbar_title: str = "Conviction",
    num_cases_to_plot: int = 5,
    title: str = "Anomalies",
    xaxis_title: str = "Feature",
    yaxis_title: t.Optional[str] = None,
) -> go.Figure:
    """
    Plot anomalous cases using a heat map which shows conviction values for each feature.

    Parameters
    ----------
    anomalies : DataFrame
        A DataFrame containing the anomalous cases to visualize.
    convictions : DataFrame
        A DataFrame containing the conviction data for the cases to visualize.
    colorbar_title : str, default "Conviction
        The title for the plot's colorbar.
    num_cases_to_plot : int, default 5
        The number of cases from ``anomalies`` to plot.
    title : str, default "Anomalies"
        The title to set for the plot.
    xaxis_title : str, default "Feature"
        The title for the figure's x axis.
    yaxis_title : str, optional
        The title for the figure's y axis.

    Returns
    -------
    Figure
        The resultant `Plotly` figure.
    """
    fig = make_subplots()

    # Use the colors from the existing red-blue colorscale to create a custom colorscale "centered"
    # around 1.0 as a neutral value.
    cmap = px.colors.diverging.RdBu
    ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    colorscale = [[0, cmap[0]], [0.2, "rgb(247, 247, 247)"], [0.4, cmap[-3]], [1.0, cmap[-1]]]

    convictions_to_plot = convictions.head(num_cases_to_plot)
    anomalies_to_plot = anomalies.head(num_cases_to_plot)

    fig.add_trace(go.Heatmap(
        z=convictions_to_plot,
        x=convictions_to_plot.columns,
        y=convictions_to_plot.index,
        xgap=3,
        ygap=3,
        text=anomalies_to_plot.astype(str),
        texttemplate="%{text}",
        hovertemplate="Conviction=%{z}",
        name="Outliers",
        coloraxis="coloraxis",
    ))

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmax=5,
            cmin=0,
            colorbar=dict(
                tickvals=ticks,
                ticktext=[str(t) for t in ticks[:-1]] + ["≥" + str(ticks[-1])],
                title=colorbar_title,
            )
        ),
        title=dict(text=title),
        yaxis=dict(autorange="reversed", title=yaxis_title),
        xaxis=dict(title=xaxis_title),
    )

    return fig


def plot_dataset(
    data: DataFrame,
    x: str,
    y: str,
    *,
    alpha: float = 1.0,
    boundary_cases: t.Optional[DataFrame] = None,
    highlight_index: t.Optional[t.Any | list[t.Any]] = None,
    highlight_label: t.Optional[str] = None,
    highlight_selection_conditions: t.Optional[dict[str, t.Any]] = None,
    hue: t.Optional[str] = None,
    most_similar_cases: t.Optional[DataFrame] = None,
    size: t.Optional[str] = None,
    title: t.Optional[str] = None,
) -> go.Figure:
    """
    Create a figure which displays an entire dataset with certain cases being specifically highlighted.

    Namely, the cases which can be specifically highlighted are:

    - A particular case index/set of indices,
    - the most similar cases of those indices,
    - and the boundary cases of those indices.

    The legend is broken into two groups, the cases and the highlighted cases, where the cases are grouped by
    the column represented by the ``hue`` parameter.

    Parameters
    ----------
    data : DataFrame
        The data to plot.
    x : str
        The feature to place on the x axis.
    y : str
        The feature to place on the y axis.
    alpha : float, default 1.0
        The transparency to assign to each marker when plotting the dataset.
    boundary_cases: DataFrame, optional
        The boundary cases to plot, if any.
    highlight_index : Any or list of Any, optional
        The index of one or more individual cases to highlight. Takes priority over
        ``highlight_selection_conditions``.
    highlight_label : str, Optional
        The label to assign to the highlighted case.
    highlight_selection_conditions : dict of str -> Any, optional
        A mapping of feature names to feature values that describes conditions for selecting case(s)
        to highlight.
    hue : str, optional
        The feature to use when determining the hue of the plotted dataset.
    most_similar_cases : DataFrame, optional
        The most similar cases to plot, if any.
    size : str, optional
        The feature to use to determine the size of the dataset markers.
    title : str, optional
        The title for the figure.

    Returns
    -------
    Figure
        The resultant `Plotly` figure.
    """
    fig = make_subplots(
        x_title=x,
        y_title=y,
    )

    if title:
        fig.update_layout(
            title=dict(text=title)
        )
    fig.update_layout(
        height=800,
        legend=dict(groupclick="toggleitem"),
    )

    primary_scatter_kwargs = dict(
        mode="markers",
        marker=dict(
            opacity=alpha,
            size=10,
            line=dict(
                color="darkgrey",
                width=2,
            ),
            sizemin=7.5,
            sizemode="area"
        ),
        legendgroup="cases",
        legendgrouptitle_text="Cases",
    )

    if size is not None:
        primary_scatter_kwargs["marker_size"] = data[size]

    if hue is not None:
        for name, group in data.groupby(hue):
            fig.add_trace(go.Scattergl(
                name=name,
                x=group[x],
                y=group[y],
                **primary_scatter_kwargs
            ))
    else:
        fig.add_trace(go.Scattergl(
            name="Cases",
            x=data[x],
            y=data[y],
            **primary_scatter_kwargs
        ))

    special_cases_kwargs = dict(
        mode="markers",
        legendgroup="special_cases",
        legendgrouptitle_text="Special Cases"
    )
    special_case_marker_kwargs = dict(
        opacity=0.75,
        size=20,
        line=dict(
            color="darkgrey",
            width=2,
        )
    )

    if most_similar_cases is not None and len(most_similar_cases):
        fig.add_trace(go.Scattergl(
            name="Most Similar Cases",
            x=most_similar_cases[x],
            y=most_similar_cases[y],
            marker=dict(
                symbol="square",
                color="grey",
                **special_case_marker_kwargs,
            ),
            **special_cases_kwargs,
        ))

    if boundary_cases is not None and len(boundary_cases):
        fig.add_trace(go.Scattergl(
            name="Boundary Cases",
            x=boundary_cases[x],
            y=boundary_cases[y],
            marker=dict(
                symbol="pentagon",
                color="brown",
                **special_case_marker_kwargs,
            ),
            **special_cases_kwargs,
        ))

    highlight_cases = None
    if highlight_index is not None:
        if not isinstance(highlight_index, list):
            highlight_index = [highlight_index]
        highlight_cases = data.loc[highlight_index, :]
    elif highlight_selection_conditions is not None:
        highlight_query = " and ".join([f"{k} == {repr(v)}" for k, v in highlight_selection_conditions.items()])
        highlight_cases = data.query(highlight_query)

    if highlight_cases is not None and len(highlight_cases):
        fig.add_trace(go.Scattergl(
            name=highlight_label,
            x=highlight_cases[x],
            y=highlight_cases[y],
            marker=dict(
                symbol="triangle-up",
                color="black",
                **special_case_marker_kwargs,
            ),
            **special_cases_kwargs,
        ))

    return fig


def plot_drift(
    df: DataFrame,
    *,
    compute_rolling_mean: bool = True,
    line_positions: list[int] = None,
    rolling_window: int = 10,
    title: str = "Model Drift \u2014 Conviction Over Time",
    xaxis_title: str = "Case Index",
    yaxis_title: str = "Conviction",
) -> go.Figure:
    """
    Plot model drift over time.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the drift measures to plot.
    compute_rolling_mean : bool, default True
        Whether to compute the rolling mean for the variable to plot.
    line_positions : list of int, optional
        Positions along the X axis to plot black lines.
    rolling_window : int, default 10
        The window to use when computing the rolling mean.
    title : str, "Model Drift \u2014 Conviction Over Time"
        The title for the figure.
    xaxis_title : str, "Case Index"
        The title for the figure's x axis.
    yaxis_title : str, "Conviction"
        The title for the figure's y axis.

    Returns
    -------
    Figure
        The resultant `Plotly` figure.
    """
    fig = make_subplots()

    if compute_rolling_mean:
        df = df.rolling(rolling_window).mean().fillna(0)

    df = pd.melt(df.reset_index(names=["time"]), id_vars="time")
    for name, group in df.groupby("variable"):
        fig.add_trace(go.Scattergl(
            x=group.time, y=group.value, name=name,
        ))

    if line_positions:
        for line_pos in line_positions:
            fig.add_vline(x=line_pos, line=dict(color="black"))

    fig.update_layout(
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title=yaxis_title),
        title=dict(text=title)
    )

    return fig


def plot_kl_divergence(
    data: Series,
    *,
    title: str = "KL Divergence Over Time",
    xaxis_title: str = "Case Index",
    yaxis_title: str = "KL Divergence",
) -> go.Figure:
    """
    Plot KL-Divergence over time.

    Parameters
    ----------
    data : Series
        A Series containing the KL-Divergence data.
    title : str, "KL Divergence Over Time"
        The title for the figure.
    xaxis_title : str, "Case Index"
        The title for the x axis.
    yaxis_title : str, "KL Divergence"
        The title for the y axis.

    Returns
    -------
    Figure
        The resultant `Plotly` figure.
    """
    non_inf_df = data[~(np.isinf(data))]
    inf_df = data[np.isinf(data)]

    non_inf_rolling_y = min(round(len(non_inf_df) / 2), 5)

    fig = make_subplots()
    fig.add_trace(go.Scattergl(
        x=non_inf_df.index,
        y=non_inf_df.rolling(non_inf_rolling_y).mean(),
        name="KL Divergence",
        hovertemplate="%{y}<extra></extra>"
    ))
    fig.add_trace(go.Scattergl(
        x=inf_df.index,
        y=[np.max(non_inf_df)] * len(inf_df),
        marker=dict(color="black"),
        name="Infinity",
        hovertemplate="∞<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=0, dtick=1.0, title=xaxis_title),
        yaxis=dict(title=yaxis_title),
        title=dict(text=title),
    )

    return fig


def plot_interpretable_prediction(
    react: Reaction,
    *,
    actual_value: t.Optional[float] = None,
    generative_reacts: t.Optional[list[float]] = None,
    residual: t.Optional[float] = None,
    secondary_yaxis_title: str = "Influence Weight",
    title: t.Optional[str] = None,
    xaxis_title: t.Optional[str] = None,
    yaxis_title: str = "Density",
) -> go.Figure | list[go.Figure]:
    """
    Plot a prediction with additional information for interpreting the result.

    Parameters
    ----------
    react : Reaction
        The reaction predicting the action feature(s) to visualize. If this contains more than one action feature,
        each will be given its own plot.
    generative_reacts : list of float, optional
        An optional list of values for the action feature to visualize. This will be used to visualize a
        KDE plot to characterize the distribution of values around the predicted and actual values. If this is None,
        the distribution of influential cases in the react will be used instead, if present.
    actual_value : float, optional
        The actual value for the point that was predicted. If this is None, only the predicted value will be
        visualized.
    residual : float, optional
        The residual for the feature that was predicted, local or global. Used to display an error bar around the
        predicted value. If this is None, no error bar will be displayed
    secondary_yaxis_title : str, default "Influence Weight"
        The title for thefigure's secondary y axis.
    title : str, optional
        The title for the figure.
    xaxis_title : str, optional
        The title for the figure's x axis. If None, the action feature will be used.
    yaxis_title : str, default "Density"
        The title for the figure's y axis.

    Returns
    -------
    Figure or list of Figure
        The resultant `Plotly` figure(s).
    """
    figures = []
    for action_feature in react["action"].columns:
        predicted_value = react["action"][action_feature].iloc[0]

        influential_cases = react.get("details", {}).get("influential_cases")
        influential_cases = influential_cases[0] if influential_cases else None

        if generative_reacts is not None:
            action_distribution = generative_reacts
        else:
            action_distribution = [c[action_feature] for c in influential_cases]
        action_kde = gaussian_kde(action_distribution)
        density_x = np.linspace(
            min(action_distribution) * 0.6,
            max(action_distribution) * 1.4,
            len(action_distribution),
        )
        density_y = action_kde(density_x)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scattergl(x=density_x, y=density_y, fill="tonexty", name="Distribution", hoverinfo="skip", mode="lines")
        )
        case_hover_template = "Value: %{x}<br />Influence: N/A"

        if influential_cases is not None:
            max_inf_weight = max([ic[".influence_weight"] for ic in influential_cases])
        else:
            max_inf_weight = None

        # Add predicted value
        error_x = {"array": [residual]} if residual is not None else None
        fig.add_trace(
            go.Scattergl(
                x=[predicted_value], y=[max_inf_weight * 1.05],
                name="Predicted Value",
                mode="markers", marker={"size": 15, "color": "purple"},
                hovertemplate=case_hover_template,
                error_x=error_x
            ),
            secondary_y=True
        )

        if actual_value is not None:
            # Add actual value
            fig.add_trace(
                go.Scattergl(
                    x=[actual_value], y=[max_inf_weight * 1.05],
                    name="Actual Value",
                    mode='markers', marker=dict(size=15, color="orange"),
                    hovertemplate=case_hover_template
                ),
                secondary_y=True
            )

        # Update axes, hover mode
        fig.update_xaxes(
            title_text=xaxis_title or action_feature,
            autorange=True
        )
        fig.update_yaxes(title_text=yaxis_title, color="blue", secondary_y=False)
        fig.update_yaxes(title_text=secondary_yaxis_title, color="green", secondary_y=True)

        if influential_cases is not None:
            inf_case_hover_template = (
                "Value: %{x}<br />"
                "Influence: %{y}<br />"
                "<br />"
                "Session: %{customdata[0]}<br />"
                "Session Index: %{customdata[1]}"
            )

            inf_case_values = []
            inf_case_weights = []
            inf_case_labels = []

            # Add influential cases
            for i in influential_cases:
                inf_case_values.append(i[action_feature])
                inf_case_weights.append(i[".influence_weight"])
                inf_case_labels.append([i[".session"], i[".session_training_index"]])

            fig.add_trace(
                go.Scattergl(
                    x=inf_case_values,
                    y=inf_case_weights,
                    name="Influential Case",
                    hovertemplate=inf_case_hover_template,
                    customdata=inf_case_labels,
                    mode="markers", marker=dict(color="green")
                ),
                secondary_y=True
            )

        if title is not None:
            fig.update_layout(title=dict(text=title))

        figures.append(fig)

    if len(figures) == 1:
        return figures[0]


def plot_fairness_disparity(
    fairness_results: dict,
    reference_class: str,
    *,
    fairness_threshold: float = 0.75,
    x_tickangle: bool | float = False,
    fair_color: str = "#9BBf85",
    unfair_color: str = "#B3589A",
    reference_class_color: str = 'grey',
):
    """
    Helper function for plotting fairness disparity results.

    Parameters
    ----------
    fairness_results : dict
        Dictionary of the fairness disparity ratios.
    reference_class : str
        The reference class for the metric calculation.
    fairness_threshold : float, default 0.75
        Threshold for values to be classified as fair. Values below this threshold are colored red when
        graphing while values above are colored green.
    x_tickangle : bool or float, default False
        Whether to rotate the x-axis labels. If False, the labels will not be rotated. If True,
        the default angle is 45 degrees. If float, then that angle of rotation will be used.
    fair_color : str, default '#9BBf85'
        The bar chart color for values that are above the ``fairness_threshold``. Accepts `Plotly` compatible
        values.
    unfair_color : str, default '#B3589A'
        The bar chart color for values that are below the ``fairness_threshold``. Accepts `Plotly` compatible
        values.
    reference_class_color : str, default 'grey'
        The bar chart color for the reference class. Accepts `Plotly` compatible
        values.

    Returns
    -------
    Figure
        The resultant `Plotly` figure.

    Examples
    --------
    .. code-block::python

        >> fairness_results
            {
                'Dataset1': {'Male': 0.8, 'Female': 0.7, 'Other': 0.9},
                'Dataset2': {'Male': 0.6, 'Female': 0.9, 'Other': 0.5},
                'Dataset3': {'Male': 0.7, 'Female': 0.4, 'Other': 0.6}
            }
        >> fig = plot_fairness_disparity(fairness_results, reference_class='Male')

    """
    num_subplots = len(fairness_results)
    fig = make_subplots(rows=1, cols=num_subplots, subplot_titles=list(fairness_results.keys()))

    for i, (_, results) in enumerate(fairness_results.items(), 1):
        sorted_data = {key: results[key] for key in sorted(results, key=lambda x: (x != reference_class, x))}
        for key, value in sorted_data.items():
            color = reference_class_color if key == reference_class else (
                unfair_color if value < fairness_threshold else fair_color
            )
            key_value = key if key != reference_class else f'{reference_class} (ref)'

            fig.add_trace(
                go.Bar(x=[key_value], y=[value], marker_color=color, name=key_value),
                row=1, col=i
            )
            fig.add_annotation(x=key_value, y=value / 2, text=str(round(value, 2)), showarrow=False,
                               xref=f"x{i}", yref=f"y{i}")

    fig.update_layout(showlegend=False)

    if x_tickangle:
        if isinstance(x_tickangle, bool):
            x_tickangle = 45
        fig.update_xaxes(tickangle=x_tickangle)

    return fig


def compose_figures(
    figures: list[go.Figure],
    rows: int,
    cols: int,
    **make_subplots_kwargs,
) -> go.Figure:
    """
    Helper function for composing several plotly `Figure`s.

    This is particularly useful for `Figure`s created by ``plotly.express``. Any
    unspecified keyword arguments are passed to ``plotly.subplots.make_subplots``.

    Parameters
    ----------
    figures : List[go.Figure]
        The `Figure`s to compose in the order to compose them.
    rows : int
        The number of rows to include in the composed `Figure`.
    cols : int
        The number of columns to include in the composed `Figure`.

    Returns
    -------
    Figure
        The composed `Figure`.
    """
    figure_trace_map: dict[tuple[int, int], list] = {}
    subplot_titles = []

    if rows < 1 or cols < 1:
        raise ValueError("Neither `rows` nor `cols` can be less than 1.")

    if rows * cols < len(figures):
        raise ValueError(f"A {rows}x{cols} grid of subplots cannot fit {len(figures)} figures.")

    # The figures to compose may be fewer than the total number of cells,
    # e.g. when  we fill a 2x2 grid with 3 subplots. The fig_counter helps
    # to break out when we are out of figures.
    fig_counter = 0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            figure_trace_map[(i, j)] = [
                t for t in figures[fig_counter]["data"]
            ]
            subplot_titles.append(figures[fig_counter]["layout"]["title"]["text"])
            fig_counter += 1

            if fig_counter == len(figures):
                break

    for k in {"rows", "cols", "subplot_titles"}:
        make_subplots_kwargs.pop(k, None)

    return_figure = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, **make_subplots_kwargs)
    for key, value in figure_trace_map.items():
        row, col = key
        return_figure.add_traces(value, rows=row, cols=col)

    # Remove duplicate entries from the legend
    legend_names = set()
    return_figure.for_each_trace(
        lambda t: t.update(showlegend=False) if t.name in legend_names else legend_names.add(t.name)
    )

    # Multiple subplots may have the same bingroup specified, so we override that here.
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            return_figure.update_traces(bingroup=f"subplot_{i},{j}", row=i, col=j)

    return return_figure


def plot_umap(
    data: DataFrame | Trainee,
    *,
    action_feature: t.Optional[str] = None,
    color: t.Optional[str] = None,
    min_dist: t.Optional[float] = None,
    n_cases: t.Optional[int] = None,
    n_neighbors: t.Optional[int] = None,
    title: str = "UMAP Representation",
    use_case_weights: bool = False,
    weight_feature: t.Optional[str] = None,
    xaxis_title: str = "Component 1",
    yaxis_title: str = "Component 2",
) -> go.Figure:
    """
    Transform data into a lower-dimensionality representation using Howso Engine and UMAP and then plot it.

    Howso Engine computes pairwise distances which are then used with UMAP's ``precomputed``
    metric.

    Parameters
    ----------
    data : DataFrame | Trainee
        The data to transform or a :class:`Trainee` containing the data to transform.
    action_feature : str, optional
        The action feature to use when selecting hyperparameters for :meth:`Trainee.get_distances`.
    color : str, optional
        The name of the column in ``data`` to use for determining marker color.
    min_dist : float, optional
        The ``min_dist`` parameter for ``umap.UMAP``. If None, this will be the :math:`p` norm
        of the feature residuals, where :math:`p` is selected by :meth:`Trainee.analyze`.
    n_cases : int, optional
        The number of cases to compute pairwise distances for. If None, then all of the cases
        are used.
    n_neighbors : int, optional
        The ``n_neighbors`` parameter for ``umap.UMAP``. If None, this will be the :math:`k`
        selected by :meth:`Trainee.analyze`.
    title : str, default "UMAP Representation"
        The title for the figure.
    use_case_weights : bool, default False
        Whether to use case weights when selecting hyperparameters for :meth:`Trainee.get_distances`.
    weight_feature: str, optional
        The weight feature to use when selecting hyperparameters for :meth:`Trainee.get_distances`.
    xaxis_title : str, default "Component 1"
        The title for the x-axis.
    yaxis_title : str, default "Component 2"
        The title for the y-axis.

    Returns
    -------
    Figure
        The resultant `Plotly` figure.
    """
    if isinstance(data, DataFrame):
        features = infer_feature_attributes(data)
        t = Trainee(features=features)
        t.train(data, skip_auto_analyze=True)
        t.analyze()
    elif isinstance(data, Trainee):
        t = data
    else:
        raise TypeError("`data` must be a Trainee or a DataFrame.")

    case_indices = None
    if n_cases is not None:
        sampled_cases = t.get_cases(
            features=[".session", ".session_training_index"] + list(t.features),
            session=t.get_sessions()[0]["id"],
        ).sample(n_cases)
        case_indices = sampled_cases[[".session", ".session_training_index"]]
        case_indices = case_indices.values.tolist()
    else:
        sampled_cases = t.get_cases(
            features=[".session", ".session_training_index"] + list(t.features),
            session=t.get_sessions()[0]["id"],
        )

    distances = t.get_distances(
        case_indices=case_indices,
        action_feature=action_feature,
        use_case_weights=use_case_weights,
        weight_feature=weight_feature,
    )["distances"]
    hyperparameter_map = t.get_params(action_feature=".targetless")["hyperparameter_map"]

    n_neighbors = n_neighbors or hyperparameter_map["k"]
    p = hyperparameter_map["p"]

    if min_dist is None:
        residuals = t.get_prediction_stats(
            action_feature=action_feature,
            details={"feature_full_residuals": True},
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )
        min_dist = float((residuals.values ** p).sum() ** (1 / p))
        min_dist = min(round(min_dist, 3), 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points = umap.UMAP(
            metric="precomputed",
            min_dist=min_dist,
            n_neighbors=n_neighbors,
        ).fit_transform(distances)

    scatter_kwargs = {}
    labels = {
        "x": xaxis_title,
        "y": yaxis_title,
    }
    if color is not None:
        scatter_kwargs["color"] = sampled_cases[color].astype(object)
        labels["color"] = color

    fig = px.scatter(
        x=points[:, 0],
        y=points[:, 1],
        title=title,
        labels=labels,
        **scatter_kwargs
    )
    return fig
