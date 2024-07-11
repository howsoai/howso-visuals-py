from __future__ import annotations

import typing as t
import warnings

import numpy.typing as npt
from pandas import DataFrame

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import umap

from howso.engine import Trainee
from howso.utilities import infer_feature_attributes


__all__ = [
    "howso_umap_transform",
]


def howso_umap_transform(
    data: DataFrame,
    *,
    min_dist: t.Optional[float] = None,
    n_neighbors: t.Optional[int] = None,
    n_cases: t.Optional[int] = None,
) -> npt.ArrayLike | tuple[npt.ArrayLike, DataFrame]:
    """
    Transform data into a lower-dimensionality representation using Howso Engine and UMAP.

    Howso Engine computes pairwise distances which are then used with UMAP's ``precomputed``
    metric.

    Parameters
    ----------
    data : DataFrame
        The data to transform.
    min_dist : Optional[float], optional
        The ``min_dist`` parameter for ``umap.UMAP``. If None, this will be the :math:`p` norm
        of the feature residuals, where :math:`p` is selected by :meth:`Trainee.analyze`.
    n_neighbors : Optional[int], optional
        The ``n_neighbors`` parameter for ``umap.UMAP``. If None, this will be the :math:`k`
        selected by :meth:`Trainee.analyze`.
    n_cases : Optional[int], optional
        The number of cases to compute pairwise distances for. If None, then all of the cases
        are used.

    Returns
    -------
    ArrayLike
        A 2-dimensional representation of the input data.
    DataFrame
        The selected cases. This is only returned if ``n_cases`` is not None.
    """
    features = infer_feature_attributes(data)
    t = Trainee(features=features)
    t.train(data, skip_auto_analyze=True)
    t.analyze()

    case_indices = None
    if n_cases is not None:
        sampled_cases = t.get_cases(
            features=[".session", ".session_training_index"] + list(features),
            session=t.get_sessions()[0]["id"],
        )
        case_indices = sampled_cases[[".session", ".session_training_index"]]
        case_indices = case_indices.values.tolist()
    
    distances = t.get_distances(case_indices=case_indices)["distances"]
    hyperparameter_map = t.get_params(action_feature=".targetless")
    
    n_neighbors = n_neighbors or hyperparameter_map["k"]
    p = hyperparameter_map["p"]

    if min_dist is None:
        residuals = t.react_aggregate(details={"feature_residuals_full": True})
        min_dist = float((residuals.values ** p).sum() ** (1 / p))
        min_dist = min(round(min_dist, 3), 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points = umap.UMAP(
            metric="precomputed",
            min_dist=min_dist,
            n_neighbors=n_neighbors,
        ).fit_transform(distances)
    
    if n_cases is not None:
        return points, sampled_cases
    else:
        return points
