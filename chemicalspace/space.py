import numpy as np
from .layers.acquisition import STRATEGIES, pick_samples
from .layers.base import T
from .layers.clustering import ChemicalSpaceClusteringLayer
from .layers.neigbors import ChemicalSpaceNeighborsLayer
from .layers.projection import PROJECTION_METHODS, project_space
from numpy.typing import NDArray


class ChemicalSpace(ChemicalSpaceClusteringLayer, ChemicalSpaceNeighborsLayer):
    def project(
        self,
        n_components: int = 2,
        method: PROJECTION_METHODS = "tsne",
        seed: int = 42,
        n_jobs: int = 1,
        **kwargs
    ) -> NDArray[np.float_]:
        return project_space(self, n_components, method, seed, n_jobs, **kwargs)

    def pick(self, n: int, strategy: STRATEGIES = "random") -> T:  # type: ignore
        pick_idx = pick_samples(
            n=n, strategy=strategy, inputs=self.features, scores=self.scores
        )
        mask = np.zeros(len(self.features), dtype=bool)
        mask[pick_idx] = True

        return self.mask(mask)
