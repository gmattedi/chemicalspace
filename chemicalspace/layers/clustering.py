import warnings
from functools import partial
from typing import Any, Generator, List, Literal, Optional, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .base import ChemicalSpaceBaseLayer
from .utils import SEED, reduce_sum

CLUSTERING_METHODS: TypeAlias = Literal["kmedoids", "agglomerative-clustering"]
CLUSTER_NUMBER: TypeAlias = int | None


def get_optimal_cluster_number(
    features: NDArray[Any],
    model: Any,
    min_clusters: int = 2,
    max_clusters: Optional[int] = None,
    metric: str = "jaccard",
    n_runs: int = 10,
) -> int:
    """
    Calculates the optimal number of clusters using the silhouette score.

    Args:
        features (ndarray): The input features for clustering.
        model (Any): The clustering model to use. Must be instantiable,
            have a `n_clusters` parameter, and a `fit_predict` method.
        min_clusters (int, optional): The minimum number of clusters to consider. Defaults to 2.
        n_runs (int, optional): The number of runs to average the silhouette score. Defaults to 10.

    Returns:
        int: The optimal number of clusters.

    """
    from sklearn.metrics import silhouette_score

    if max_clusters is None:
        max_clusters = len(features) // 2

    scan_n = np.linspace(min_clusters, max_clusters, n_runs, dtype=int)
    scan_n = np.unique(scan_n)

    scores: List[float] = []
    for n_clusters in scan_n:
        model_inst = model(n_clusters=n_clusters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = model_inst.fit_predict(features)
            score = silhouette_score(features, labels, metric=metric)
        scores.append(float(score))

    n = scan_n[np.argmax(scores)]

    return n


# @dataclass(frozen=False, repr=False)
class ChemicalSpaceClusteringLayer(ChemicalSpaceBaseLayer):
    """
    A class representing a layer for clustering chemical space data.

    Inherits from ChemicalSpaceBaseLayer.

    Methods:
        cluster: Perform clustering on the chemical space data.
        ksplits: Generate train-test splits based on clustering.

    Attributes:
        Inherits attributes from ChemicalSpaceBaseLayer.
    """

    def cluster(
        self,
        n_clusters: CLUSTER_NUMBER = None,
        method: CLUSTERING_METHODS = "kmedoids",
        seed: int = SEED,
        **kwargs,
    ) -> NDArray[np.int_]:
        """
        Perform clustering on the chemical space data.

        Args:
            n_clusters (int | None): The number of clusters to create.
                If None, the number of clusters will be determined by silhouette score
            method (str): The clustering method to use.
            seed (int, optional): The random seed for reproducibility. Defaults to SEED.
            **kwargs: Additional keyword arguments to pass to the clustering algorithm.

        Returns:
            NDArray[np.int_]: An array of cluster labels for each molecule.

        """
        if method == "kmedoids":
            from sklearn_extra.cluster import KMedoids

            obj = partial(KMedoids, metric="jaccard", random_state=seed)

        elif method == "agglomerative-clustering":
            from sklearn.cluster import AgglomerativeClustering

            obj = partial(AgglomerativeClustering, metric="jaccard", linkage="complete")

        else:
            raise ValueError(f"Invalid clustering method: {method}")

        if n_clusters is None:
            n_clusters = get_optimal_cluster_number(self.features, obj, **kwargs)

        clusterer = obj(n_clusters, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = np.array(clusterer.fit_predict(self.features), dtype=int)

        return labels

    def yield_clusters(
        self,
        n_clusters: CLUSTER_NUMBER = None,
        method: CLUSTERING_METHODS = "kmedoids",
        seed: int = SEED,
        **kwargs,
    ) -> Generator[ChemicalSpaceBaseLayer, Any, None]:
        """
        Yields clusters from the chemical space.

        Args:
            n_clusters (int | None): The number of clusters to create.
                If None, the number of clusters will be determined by silhouette score
            method (CLUSTERING_METHODS): The clustering method to use.
            seed (int, optional): The random seed for reproducibility. Defaults to SEED.
            **kwargs: Additional keyword arguments to be passed to the clustering method.

        Yields:
            Generator[ChemicalSpaceBaseLayer, Any, None]: A generator that yields each
                cluster as a ChemicalSpaceBaseLayer object.
        """
        labels = self.cluster(n_clusters, method, seed, **kwargs)

        n = len(set(labels))

        for i in range(n):
            mask = labels == i
            yield self.mask(mask)

    def ksplits(
        self,
        n_splits: int,
        method: CLUSTERING_METHODS = "kmedoids",
        seed: int = SEED,
        **kwargs,
    ) -> Generator[
        Tuple["ChemicalSpaceBaseLayer", "ChemicalSpaceBaseLayer"], None, None
    ]:
        """
        Generate train-test splits based on clustering.

        Args:
            n_splits (int): The number of splits to generate.
            method (str): The clustering method to use.
            seed (int, optional): The random seed for reproducibility. Defaults to SEED.
            **kwargs: Additional keyword arguments to pass to the clustering algorithm.

        Yields:
            A tuple of ChemicalSpaceBaseLayer objects representing the train and test splits.

        """
        clusters = list(self.yield_clusters(n_splits, method, seed, **kwargs))

        for i in range(n_splits):
            train_lst: List[ChemicalSpaceBaseLayer] = []
            test = clusters[i]

            for j in range(n_splits):
                if j != i:
                    train_lst.append(clusters[j])

            train: ChemicalSpaceBaseLayer = reduce_sum(*train_lst)

            yield train, test
