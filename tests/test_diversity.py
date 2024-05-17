import os

import numpy as np
import pytest

from chemicalspace.layers.diversity import (
    DIVERSITY_METHODS,
    diversity_methods_dict,
    ChemicalSpaceDiversityLayer,
)

np.random.seed(42)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILE = os.path.join(TESTS_DIR, "data", "inputs1.smi")


@pytest.fixture
def space() -> ChemicalSpaceDiversityLayer:
    return ChemicalSpaceDiversityLayer.from_smi(INPUT_SMI_FILE)


def test_uniqueness(space: ChemicalSpaceDiversityLayer) -> None:
    uniqueness = space.uniqueness()
    assert isinstance(uniqueness, float)
    assert 0 <= uniqueness <= 1


@pytest.mark.parametrize("method", tuple(diversity_methods_dict.keys()))
def test_diversity(
    space: ChemicalSpaceDiversityLayer, method: DIVERSITY_METHODS
) -> None:
    diversity = space.diversity(method=method)
    assert isinstance(diversity, float)
    assert 0 <= diversity <= 1
