import os

import numpy as np
import pytest

from chemicalspace.layers.projection import (
    PROJECTION_METHODS,
    ChemicalSpaceProjectionLayer,
)

np.random.seed(42)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILE = os.path.join(TESTS_DIR, "data", "inputs1.smi")


@pytest.fixture
def space() -> ChemicalSpaceProjectionLayer:
    return ChemicalSpaceProjectionLayer.from_smi(INPUT_SMI_FILE)


@pytest.mark.parametrize("n_components", (2,))
@pytest.mark.parametrize("method", ("pca", "tsne"))  # umap is slow
def test_project(
    space: ChemicalSpaceProjectionLayer, n_components: int, method: PROJECTION_METHODS
) -> None:
    if method == "tsne":
        kwargs = {"perplexity": 3}
    elif method == "umap":
        kwargs = {"n_neighbors": 3}
    else:
        kwargs = {}

    proj = space.project(
        n_components=n_components, method=method, metric="jaccard", n_jobs=1, **kwargs
    )
    assert isinstance(proj, np.ndarray)
    assert proj.shape == (10, n_components)
