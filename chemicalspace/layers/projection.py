from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .base import ChemicalSpaceBaseLayer
from .utils import SEED

PROJECTION_METHODS: TypeAlias = Literal["umap", "tsne", "pca"]


def project_space(
    chemical_space: ChemicalSpaceBaseLayer,
    n_components: int = 2,
    method: PROJECTION_METHODS = "tsne",
    metric: str = "jaccard",
    seed: int = SEED,
    n_jobs: int = 1,
    **kwargs,
) -> NDArray[np.float_]:
    """
    Project the chemical space data to a lower-dimensional space.

    Args:
        chemical_space (ChemicalSpaceBaseLayer): The chemical space layer to project.
        n_components (int, optional): The number of components in the lower-dimensional space. Defaults to 2.
        method (str, optional): The projection method to use. Defaults to "tsne".
        metric (str, optional): The metric to use for the projection. Defaults to "jaccard".
        seed (int, optional): The random seed for reproducibility. Defaults to SEED.
        n_jobs (int, optional): The number of parallel jobs to run. Defaults to 1.
        **kwargs: Additional keyword arguments to pass to the projection algorithm.

    Returns:
        NDArray[np.float_]: An array of the projected data in the lower-dimensional space.

    """
    if method == "umap":
        from umap import UMAP

        projector = UMAP(
            n_components=n_components,
            random_state=seed,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE

        projector = TSNE(
            n_components=n_components,
            random_state=seed,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif method == "pca":
        from sklearn.decomposition import PCA

        projector = PCA(n_components=n_components, random_state=seed, **kwargs)
    else:
        raise ValueError(f"Invalid projection method: {method}")

    proj = np.array(projector.fit_transform(chemical_space.features))

    return proj


class ChemicalSpaceProjectionLayer(ChemicalSpaceBaseLayer):
    """
    A layer for projecting chemical space data to a lower-dimensional space.
    """

    def project(
        self,
        n_components: int = 2,
        method: PROJECTION_METHODS = "tsne",
        seed: int = 42,
        n_jobs: int = -1,
        **method_kwargs,
    ) -> NDArray[np.float_]:
        """
        Project the chemical space data to a lower-dimensional space.
        Args:
            n_components (int, optional): The number of components in the lower-dimensional space. Defaults to 2.
            method (str, optional): The projection method to use. Defaults to "tsne".
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
            n_jobs (int, optional): The number of parallel jobs to run. Defaults to -1.
            **method_kwargs: Additional keyword arguments to pass to the projection algorithm.

        Returns:

        """
        return project_space(
            self,
            n_components,
            method,
            metric=self.metric,
            seed=seed,
            n_jobs=n_jobs,
            **method_kwargs,
        )
