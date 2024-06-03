from typing import Optional, Callable, Dict

import numpy as np
import scipy
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances_chunked, pairwise_distances
from typing_extensions import Literal, TypeAlias

from chemicalspace.utils import ArrayIntOrFloat, parallel_map
from .base import ChemicalSpaceBaseLayer


def reduce_sum(x, _):
    return x.sum(axis=1)


def internal_distance(
    X: ArrayIntOrFloat,
    metric: str,
    n_jobs: Optional[int] = None,
    working_memory: Optional[int] = None,
) -> float:
    """
    Compute the average pairwise distance between all points in `X`,
    given a distance metric.

    The calculation is done in chunks, and parallelized.

    See `sklearn.metrics.pairwise_distances_chunked` for more details.

    Args:
        X (Union[NDArray[int], NDArray[float]]): The input data.
        metric (str): The distance metric to use.
        n_jobs (Optional[int], optional): The number of jobs to run in parallel.
            Defaults to None.
        working_memory (Optional[int], optional): The amount of memory to use for
            the computation. Defaults to None.

    Returns:
        float: The average pairwise distance between all points in `X`.

    """

    n = X.shape[0] ** 2
    total = 0

    for chunk in pairwise_distances_chunked(
        X, metric=metric, n_jobs=n_jobs, working_memory=working_memory
    ):
        total += chunk.sum()  # type: ignore

    return total / n


def vendi_score(
    X: ArrayIntOrFloat,
    metric: str,
    n_jobs: Optional[int] = None,
) -> float:
    """
    Compute the Vendi score for an array of points,
    normalized for the number of data points.

    The Vendi score is a diversity evaluation metric for machine learning models.
    It is based on the entropy of the eigenvalues of the pairwise distance matrix.

    From Friedman et al.
    "The Vendi Score: A Diversity Evaluation Metric for Machine Learning"
    https://arxiv.org/abs/2210.02410

    Args:
        X (Union[NDArray[int], NDArray[float]]): The input data.
        metric (str): The distance metric to use.
        n_jobs (Optional[int], optional): The number of jobs to run in parallel.
            Defaults to None.

    Returns:
        float: The Vendi score for the input data.

    """

    pairwise = pairwise_distances(X, metric=metric, n_jobs=n_jobs)
    pairwise_scaled = pairwise / X.shape[0]

    evals: NDArray[np.float_] = scipy.linalg.eigvalsh(pairwise_scaled)
    evals_nonzero = evals[evals > 0]

    entropy = -(evals_nonzero * np.log(evals_nonzero)).sum()
    score = np.exp(entropy)

    score_per_point = score / X.shape[0]

    return score_per_point


DIVERSITY_METHODS: TypeAlias = Literal["internal-distance", "vendi"]
diversity_methods_dict: Dict[str, Callable] = {
    "internal-distance": internal_distance,
    "vendi": vendi_score,
}


class ChemicalSpaceDiversityLayer(ChemicalSpaceBaseLayer):

    def diversity(
        self, method: DIVERSITY_METHODS = "internal-distance", **kwargs
    ) -> float:
        """
        Compute the diversity of the chemical space data.

        Args:
            method (DIVERSITY_METHODS): The diversity algorithm to use.
                Defaults to 'internal-distance'.
            **kwargs: Additional keyword arguments to pass to the diversity algorithm.

        Returns:
            float: The diversity of the chemical space data.

        """

        if method not in diversity_methods_dict:
            raise ValueError(
                f"Unknown diversity method: {method}. "
                f"Allowed methods: {list(diversity_methods_dict.keys())}"
            )

        return diversity_methods_dict[method](self.features, self.metric, **kwargs)

    def uniqueness(self) -> float:
        """
        Return the fraction of unique molecules in the chemical space.

        Returns:
            float: The fraction of unique molecules in the chemical space.
        """

        n_unique = len(set(parallel_map(self.hash_mol, self.mols)))
        return n_unique / len(self)
