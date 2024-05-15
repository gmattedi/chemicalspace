import os

import numpy as np
import pytest

from chemicalspace.layers.acquisition import (
    ChemicalSpaceAcquisitionLayer,
    strategies_dict,
    STRATEGIES,
)

np.random.seed(42)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILE = os.path.join(TESTS_DIR, "data", "inputs1.smi")


@pytest.fixture
def space() -> ChemicalSpaceAcquisitionLayer:
    cs = ChemicalSpaceAcquisitionLayer.from_smi(INPUT_SMI_FILE)
    cs.scores = tuple(range(len(cs)))  # assign scores
    return cs


@pytest.mark.parametrize("strategy", tuple(strategies_dict.keys()))
@pytest.mark.parametrize("n", (1, 5, 10))
def test_pick(
    space: ChemicalSpaceAcquisitionLayer, strategy: STRATEGIES, n: int
) -> None:
    space_picked: ChemicalSpaceAcquisitionLayer = space.pick(n=n, strategy=strategy)

    assert isinstance(space_picked, ChemicalSpaceAcquisitionLayer)
    assert len(space_picked) == n
    assert len(space - space_picked) == len(space) - n
