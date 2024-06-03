import os

import numpy as np
import pytest

from chemicalspace.layers.neighbors import ChemicalSpaceNeighborsLayer

np.random.seed(42)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILES = [
    os.path.join(TESTS_DIR, "data", name) for name in ["inputs1.smi", "inputs2.smi.gz"]
]


@pytest.fixture
def space() -> ChemicalSpaceNeighborsLayer:
    return ChemicalSpaceNeighborsLayer.from_smi(INPUT_SMI_FILES[0])


@pytest.fixture
def other_space() -> ChemicalSpaceNeighborsLayer:
    return ChemicalSpaceNeighborsLayer.from_smi(INPUT_SMI_FILES[1])


def test_overlap(space, other_space, radius=0.4, min_neighbors=1):
    indices = space.find_overlap(
        other_space, radius=radius, min_neighbors=min_neighbors
    )
    assert isinstance(indices, np.ndarray)
    assert len(indices) == 5


def test_carve(space, other_space, radius=0.4, min_neighbors=1):
    other_space_carved = other_space.carve(
        space, radius=radius, min_neighbors=min_neighbors
    )
    assert len(other_space - other_space_carved) == 5
