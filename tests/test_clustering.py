import os
from typing import Optional

import numpy as np
import pytest

from chemicalspace.layers.clustering import (
    CLUSTERING_METHODS,
    ChemicalSpaceClusteringLayer,
    get_optimal_cluster_number,
    CLUSTERING_METHODS_N,
)

np.random.seed(42)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILE = os.path.join(TESTS_DIR, "data", "inputs1.smi")


@pytest.fixture
def space() -> ChemicalSpaceClusteringLayer:
    return ChemicalSpaceClusteringLayer.from_smi(INPUT_SMI_FILE)


@pytest.mark.parametrize("method", ["kmedoids", "agglomerative-clustering"])
def test_silhouette(
    space: ChemicalSpaceClusteringLayer, method: CLUSTERING_METHODS_N
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
        assert n_clusters == 3
    elif method == "agglomerative-clustering":
        assert n_clusters == 5
    else:
        raise ValueError(f"Invalid clustering method: {method}")


@pytest.mark.parametrize(
    "method, n_clusters", [("kmedoids", None), ("agglomerative-clustering", 3)]
)
def test_cluster_n(
    space: ChemicalSpaceClusteringLayer,
    method: CLUSTERING_METHODS_N,
    n_clusters: Optional[int],
) -> None:

    clusters = space.cluster(n_clusters=n_clusters, method=method, seed=42)

    assert isinstance(clusters, np.ndarray)
    assert np.issubdtype(clusters.dtype, np.integer)
    assert len(clusters) == len(space)

    if n_clusters is not None:
        assert set(clusters) == set(range(n_clusters))


@pytest.mark.parametrize("method, kwargs", [("sphere-exclusion", {"radius": 0.8})])
def test_cluster(
    space: ChemicalSpaceClusteringLayer, method: CLUSTERING_METHODS, kwargs
) -> None:

    clusters = space.cluster(method=method, seed=42, **kwargs)

    assert isinstance(clusters, np.ndarray)
    assert np.issubdtype(clusters.dtype, np.integer)
    assert len(clusters) == len(space)


@pytest.mark.parametrize(
    "method, n_clusters", [("kmedoids", 3), ("agglomerative-clustering", None)]
)
def test_yield_clusters_n(
    space: ChemicalSpaceClusteringLayer,
    method: CLUSTERING_METHODS_N,
    n_clusters: Optional[int],
) -> None:
    clusters = space.yield_clusters(n_clusters=n_clusters, method=method, seed=42)

    i = None
    for i, cluster in enumerate(clusters):
        assert isinstance(cluster, ChemicalSpaceClusteringLayer)
        assert len(cluster) > 0

    if n_clusters is not None:
        assert i == n_clusters - 1


@pytest.mark.parametrize("method, kwargs", [("sphere-exclusion", {"radius": 0.8})])
def test_yield_clusters(
    space: ChemicalSpaceClusteringLayer, method: CLUSTERING_METHODS, kwargs
) -> None:
    clusters = space.yield_clusters(method=method, seed=42, **kwargs)

    i = None
    for i, cluster in enumerate(clusters):
        assert isinstance(cluster, ChemicalSpaceClusteringLayer)
        assert len(cluster) > 0

    assert i is not None


@pytest.mark.parametrize(
    "method, n_clusters", [("kmedoids", 3), ("agglomerative-clustering", 3)]
)
def test_ksplits(
    space: ChemicalSpaceClusteringLayer,
    method: CLUSTERING_METHODS_N,
    n_clusters: int,
) -> None:
    ks = space.ksplits(n_splits=n_clusters, method=method, seed=42)

    for cluster_train, cluster_test in ks:

        assert isinstance(cluster_train, ChemicalSpaceClusteringLayer)
        assert isinstance(cluster_test, ChemicalSpaceClusteringLayer)

        assert len(cluster_train) > 0
        assert len(cluster_test) > 0
        # Check that there's no overlap between train and test clusters
        assert cluster_train - cluster_test == cluster_train
        assert cluster_train + cluster_test == space
