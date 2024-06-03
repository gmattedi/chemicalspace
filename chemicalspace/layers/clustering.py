import warnings
from abc import ABC, abstractmethod
from functools import lru_cache, partial
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    get_args,
)

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem import Mol  # type: ignore
from typing_extensions import TypeAlias

from chemicalspace.utils import SEED, hash_mol

from .base import ChemicalSpaceBaseLayer, T

ClusteringMethodsType: TypeAlias = Literal[
    "kmedoids", "agglomerative-clustering", "sphere-exclusion", "scaffold"
]
CLUSTERING_METHODS: Tuple[str, ...] = get_args(ClusteringMethodsType)

ClusteringMethodsTypeN: TypeAlias = Literal[
    "kmedoids", "agglomerative-clustering"
]  # methods that require n_clusters
CLUSTERING_METHODS_N: Tuple[str, ...] = get_args(ClusteringMethodsTypeN)


class BaseClusteringX(ABC):
    """
    BaseClusteringMethod is an abstract class that defines the
    interface for clustering methods that take in an array of features and return an
    array of cluster labels.
    """

    @abstractmethod
    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:
        """
        Perform clustering on the input data.

        Args:
            X (ndarray): The input data to cluster.

        Returns:
            NDArray[int]: An array of cluster labels for each point.
        """
        raise NotImplementedError


class BaseClusteringMols(ABC):
    """
    BaseClusteringMols is an abstract class that defines the
    interface for clustering methods that take in an array of RDKit molecules and
    return an array of cluster labels.
    """

    @abstractmethod
    def fit_predict(self, mols: Sequence[Mol], **kwargs) -> NDArray[np.int_]:
        """
        Perform clustering on the input data.

        Args:
            mols (Sequence[Mol]): The input molecules to cluster.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArray[int]: An array of cluster labels for each molecule.
        """
        raise NotImplementedError


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
        min_clusters (int, optional): The minimum number of clusters to consider.
            Defaults to 2.
        max_clusters (int, optional): The maximum number of clusters to consider.
        metric (str, optional): The metric to use for silhouette score.
            Defaults to "jaccard".
        n_runs (int, optional): The number of runs to average the silhouette score.
            Defaults to 10.

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

    n = int(scan_n[np.argmax(scores)])

    return n


class SphereExclusion(BaseClusteringX):
    """
    A class representing a clustering algorithm based on sphere exclusion.
    It clusters input points in a set of clusters such that the minimum
    distance between any two clusters is (approximately) greater than a given radius.
    """

    def __init__(self, radius: float = 0.4, metric: str = "jaccard", **kwargs):
        """
        Initialize the SphereExclusion clustering algorithm.

        Args:
            radius (float, optional): The radius of the sphere. Defaults to 0.4.
            metric (str, optional): The metric to use for clustering.
                Defaults to "jaccard".
            **kwargs: Implemented for compatibility. Ignored.
        """
        self.radius = radius
        self.metric = metric
        _ = kwargs  # discard

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:
        """
        Perform clustering on the input data.

        Args:
            X (ndarray): The input data to cluster.

        Returns:
            NDArray[int]: An array of cluster labels for each point.
        """

        from sklearn.neighbors import BallTree

        tree = BallTree(X, metric=self.metric)

        cluster_idx = 0

        labels = np.full(X.shape[0], fill_value=-1, dtype=int)

        for i in range(X.shape[0]):
            if labels[i] != -1:
                continue

            idx = tree.query_radius(
                X[i : i + 1],
                r=self.radius,
                return_distance=False,
            )[0]

            neighbor_labels: Set[int] = set(labels[idx]) - {-1}
            for label in neighbor_labels:
                mask = labels == label
                labels[mask] = cluster_idx

            labels[idx] = cluster_idx
            cluster_idx += 1

        # Renumber the clusters to start from 0
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            labels[labels == label] = i

        return labels


class ScaffoldClustering(BaseClusteringMols):
    """
    A class representing a clustering algorithm based on Murcko scaffolds.
    """

    def __init__(self, generic: bool = True, **kwargs):
        """
        Initialize the ScaffoldClustering algorithm.

        Args:
            generic (bool, optional): Whether to use generic scaffolds. Defaults to True
            **kwargs: Implemented for compatibility. Ignored
        """
        self.generic = generic
        _ = kwargs

    def fit_predict(
        self, mols: Union[Sequence[Mol], NDArray[Mol]], **kwargs  # type: ignore
    ) -> NDArray[np.int_]:
        """
        Perform clustering on the input data.

        Args:
            mols (Sequence[Mol] | NDArray[Mol]): The input molecules to cluster.
            **kwargs: Additional keyword arguments. Ignored.

        Returns:
            NDArray[int]: Cluster labels

        """
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

        _ = kwargs  # discard

        labels: List[int] = []
        cluster_mapping: Dict[str, int] = {}
        c = 0
        for mol in mols:
            murcko = MurckoScaffold.GetScaffoldForMol(mol)
            if self.generic:
                murcko = MurckoScaffold.MakeScaffoldGeneric(murcko)

            if murcko.GetNumAtoms() > 0:
                murcko_hash = hash_mol(murcko)
            else:
                murcko_hash = ""

            if murcko_hash in cluster_mapping:
                murcko_label = cluster_mapping[murcko_hash]
            else:
                cluster_mapping[murcko_hash] = c
                murcko_label = c
                c += 1

            labels.append(murcko_label)

        return np.array(labels)


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

    @lru_cache
    def cluster(
        self,
        n_clusters: Optional[int] = None,
        method: ClusteringMethodsType = "kmedoids",
        seed: int = SEED,
        **kwargs,
    ) -> NDArray[np.int_]:
        """
        Perform clustering on the chemical space data.

        Args:
            n_clusters (int | None): The number of clusters to create
                (Ignored if not used). If None, the number of clusters will be
                determined by silhouette score
            method (str): The clustering method to use.
            seed (int, optional): The random seed for reproducibility. Defaults to SEED.
            **kwargs: Additional keyword arguments to pass to the clustering algorithm.

        Returns:
            NDArray[int]: An array of cluster labels for each molecule.

        """
        if method == "kmedoids":
            from sklearn_extra.cluster import KMedoids

            obj = partial(KMedoids, metric=self.metric, random_state=seed)

            if n_clusters is None:
                n_clusters = get_optimal_cluster_number(self.features, obj, **kwargs)

        elif method == "agglomerative-clustering":
            from sklearn.cluster import AgglomerativeClustering

            obj = partial(
                AgglomerativeClustering, metric=self.metric, linkage="complete"
            )

            if n_clusters is None:
                n_clusters = get_optimal_cluster_number(self.features, obj, **kwargs)

        elif method == "sphere-exclusion":
            obj = partial(SphereExclusion, metric=self.metric, **kwargs)
            n_clusters = -1

        elif method == "scaffold":
            obj = partial(ScaffoldClustering, **kwargs)
            n_clusters = -1

        else:
            raise ValueError(f"Invalid clustering method: {method}")

        clusterer = obj(n_clusters=n_clusters, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(clusterer, BaseClusteringMols):
                labels = np.array(clusterer.fit_predict(self.mols), dtype=int)
            else:
                labels = np.array(clusterer.fit_predict(self.features), dtype=int)

        n_clusters_actual = len(set(labels))

        if n_clusters is not None:
            if (n_clusters > 0) and (n_clusters_actual != n_clusters):
                raise ValueError(
                    f"Number of clusters ({len(set(labels))}) does not "
                    f"match number of clusters requested ({n_clusters})."
                )

        return labels

    def yield_clusters(
        self,
        n_clusters: Optional[int] = None,
        method: ClusteringMethodsType = "kmedoids",
        seed: int = SEED,
        **kwargs,
    ) -> Generator[T, Any, None]:  # type: ignore
        """
        Yields clusters from the chemical space.

        Args:
            n_clusters (int | None): The number of clusters to create
                (Ignored if not used). If None, the number of clusters will be
                determined by silhouette score
            method (CLUSTERING_METHODS): The clustering method to use
            seed (int, optional): The random seed for reproducibility. Defaults to SEED
            **kwargs: Additional keyword arguments to be passed to the clustering method

        Yields:
            Generator[ChemicalSpaceBaseLayer, Any, None]: A generator that yields each
                cluster as a ChemicalSpaceBaseLayer object.
        """
        labels = self.cluster(
            n_clusters=n_clusters,
            method=method,
            seed=seed,
            **kwargs,
        )

        n_cluster_actual = len(set(labels))

        if (n_clusters is not None) and (n_cluster_actual != n_clusters):
            raise ValueError(
                f"Number of clusters ({n_cluster_actual}) does not match number "
                f"of clusters requested ({n_clusters})."
            )

        for i in range(n_cluster_actual):
            mask = np.array(labels == i)
            yield self[mask]

    def split(
        self,
        n_splits: Optional[int] = None,
        method: ClusteringMethodsType = "kmedoids",
        seed: int = SEED,
        **kwargs,
    ) -> Generator[Tuple[T, T], None, None]:
        """
        Generate train-test splits based on clustering.

        Args:
            n_splits (Optional[int]): The number of splits to generate,
                if supported by the clustering method.
            method (str): The clustering method to use.
            seed (int, optional): The random seed for reproducibility. Defaults to SEED.
            **kwargs: Additional keyword arguments to pass to the clustering algorithm.

        Yields:
            A tuple of ChemicalSpaceBaseLayer objects representing
                the train and test splits.

        """
        labels = self.cluster(
            n_clusters=n_splits,
            method=method,
            seed=seed,
            **kwargs,
        )

        n_cluster_actual = len(set(labels))

        if (n_splits is not None) and (n_cluster_actual != n_splits):
            raise ValueError(
                f"Number of clusters ({n_cluster_actual}) does not match number of "
                f"clusters requested ({n_splits})."
            )

        n_splits_actual = len(set(labels))

        for i in range(n_splits_actual):
            mask = np.array(labels == i)
            train: T = self[~mask]
            test: T = self[mask]
            yield train, test
