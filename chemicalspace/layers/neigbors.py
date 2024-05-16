import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import BallTree

from .base import ChemicalSpaceBaseLayer, T


def _find_overlap(
    cs1: ChemicalSpaceBaseLayer,
    cs2: ChemicalSpaceBaseLayer,
    radius: float = 0.4,
    min_neighbors: int = 1,
    metric: str = "jaccard",
) -> NDArray[np.int_]:
    """
    Find the indices of points in `cs2` that have at least `min_neighbors`
    neighbors in `cs1` within a given `radius` distance.

    Parameters:
        cs1 (ChemicalSpaceBaseLayer): The first chemical space layer.
        cs2 (ChemicalSpaceBaseLayer): The second chemical space layer.
        radius (float, optional): The radius within which to search for neighbors. Defaults to 0.4.
        min_neighbors (int, optional): The minimum number of neighbors required for a point in `cs2`
            to be considered overlapping with `cs1`. Defaults to 1.
        metric (str, optional): The metric to use for the BallTree. Defaults to "jaccard".

    Returns:
        NDArray[np.int_]: The indices of points in `cs2` that have at least `min_neighbors` neighbors in `cs1`.
    """

    tree = BallTree(cs1.features, metric=metric)
    # Array of neighbor counts for each point in cs2
    num_neighbors: NDArray[np.int_] = np.array(
        tree.query_radius(
            cs2.features, r=radius, return_distance=False, count_only=True
        )
    )

    # Indices of points in cs2 that have at least `min_neighbors` neighbors in cs1
    idx_with_neighbors = np.where(num_neighbors >= min_neighbors)[0]

    return idx_with_neighbors


class ChemicalSpaceNeighborsLayer(ChemicalSpaceBaseLayer):

    def find_overlap(
        self, other: T, radius: float = 0.6, min_neighbors: int = 1  # type: ignore
    ) -> NDArray[np.int_]:
        """
        Find the indices of points in `self` that have at least `min_neighbors` neighbors
        in `other` within a given `radius` similarity threshold.

        Args:
            other (ChemicalSpaceBaseLayer): The other chemical space layer.
            radius (float, optional): The radius within which to search for neighbors. Defaults to 0.6.
            min_neighbors (int, optional): The minimum number of neighbors required for a point in `other`

        Returns:
            NDArray[np.int_]: The indices of points in `other` that have at least `min_neighbors` neighbors in `self`.

        """
        return _find_overlap(other, self, 1 - radius, min_neighbors)

    def carve(self, other: T, radius: float = 0.6, min_neighbors: int = 1) -> T:  # type: ignore
        """
        Remove points from `self` that have at least `min_neighbors` neighbors in
        `other` within a given `radius` similarity threshold.

        Args:
            other (ChemicalSpaceBaseLayer): The other chemical space layer.
            radius (float, optional): The radius within which to search for neighbors. Defaults to 0.6.
            min_neighbors (int, optional): The minimum number of neighbors required for a point in `other`

        Returns:
            T: A new chemical space layer with points removed.

        """
        idx = self.find_overlap(other, 1 - radius, min_neighbors=min_neighbors)
        mask = np.ones(len(self.features), dtype=bool)
        mask[idx] = False
        return self.mask(mask)
