import os
from typing import Optional

import pytest

from chemicalspace.layers.clustering import (
    CLUSTERING_METHODS,
    ChemicalSpaceClusteringLayer,
    get_optimal_cluster_number,
)
import numpy as np

np.random.seed(42)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILE = os.path.join(TESTS_DIR, "data", "inputs1.smi")


@pytest.fixture
def space() -> ChemicalSpaceClusteringLayer:
    return ChemicalSpaceClusteringLayer.from_smi(INPUT_SMI_FILE)


@pytest.mark.parametrize("method", ["kmedoids", "agglomerative-clustering"])
def test_silhouette(
    space: ChemicalSpaceClusteringLayer, method: CLUSTERING_METHODS
) -> None:
    from functools import partial

    from sklearn.cluster import AgglomerativeClustering
    from sklearn_extra.cluster import KMedoids

    if method == "kmedoids":
        obj = partial(KMedoids, metric="jaccard", random_state=42)
    elif method == "agglomerative-clustering":
        obj = partial(AgglomerativeClustering, metric="jaccard", linkage="complete")
    else:
        raise ValueError(f"Invalid clustering method: {method}")

    n_clusters = get_optimal_cluster_number(space.features, model=obj)

    if method == "kmedoids":
        assert n_clusters == 4
    elif method == "agglomerative-clustering":
        assert n_clusters == 5
    else:
        raise ValueError(f"Invalid clustering method: {method}")


@pytest.mark.parametrize(
    "method, n_clusters", [("kmedoids", None), ("agglomerative-clustering", 3)]
)
def test_cluster(
    space: ChemicalSpaceClusteringLayer,
    method: CLUSTERING_METHODS,
    n_clusters: Optional[int],
) -> None:

    clusters = space.cluster(n_clusters=n_clusters, method=method, seed=42)

    assert isinstance(clusters, np.ndarray)
    assert np.issubdtype(clusters.dtype, np.integer)
    assert len(clusters) == len(space)

    if n_clusters is not None:
        assert set(clusters) == set(range(n_clusters))


@pytest.mark.parametrize(
    "method, n_clusters", [("kmedoids", 3), ("agglomerative-clustering", None)]
)
def test_yield_clusters(
    space: ChemicalSpaceClusteringLayer,
    method: CLUSTERING_METHODS,
    n_clusters: Optional[int],
) -> None:
    clusters = space.yield_clusters(n_clusters=n_clusters, method=method, seed=42)

    i = None
    for i, cluster in enumerate(clusters):
        assert isinstance(cluster, ChemicalSpaceClusteringLayer)
        assert len(cluster) > 0

    if n_clusters is not None:
        assert i == n_clusters - 1


@pytest.mark.parametrize(
    "method, n_clusters", [("kmedoids", 3), ("agglomerative-clustering", 3)]
)
def test_ksplits(
    space: ChemicalSpaceClusteringLayer,
    method: CLUSTERING_METHODS,
    n_clusters: int,
) -> None:
    ks = space.ksplits(n_splits=n_clusters, method=method, seed=42)

    for train_cluster, test_cluster in ks:

        assert isinstance(train_cluster, ChemicalSpaceClusteringLayer)
        assert isinstance(test_cluster, ChemicalSpaceClusteringLayer)

        assert len(train_cluster) > 0
        assert len(test_cluster) > 0
        # Check that there-s not overlap between train and test clusters
        assert train_cluster - test_cluster == train_cluster
        assert train_cluster + test_cluster == space
