import os

import pytest

from chemicalspace import ChemicalSpace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILE = os.path.join(TESTS_DIR, "data", "inputs1.smi")


@pytest.fixture
def space() -> ChemicalSpace:
    return ChemicalSpace.from_smi(INPUT_SMI_FILE)


# Here we just test that ChemicalSpace is a subclass of the right layers
def test_attributes(space):

    assert hasattr(space, "mols")
    assert hasattr(space, "indices")
    assert hasattr(space, "scores")
    assert space._features is None

    # Base
    assert hasattr(space, "chunks")
    # Clustering
    assert hasattr(space, "cluster")
    # Neighbors
    assert hasattr(space, "carve")
    # Projection
    assert hasattr(space, "project")
    # Acquisition
    assert hasattr(space, "pick")
