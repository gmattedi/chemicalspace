import os
from pathlib import Path
from types import GeneratorType
from typing import Generator

import pytest
from rdkit.Chem import Mol

from chemicalspace.layers.base import ChemicalSpaceBaseLayer

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILES = [
    os.path.join(TESTS_DIR, "data", name) for name in ["inputs1.smi", "inputs2.smi"]
]
INPUT_SDF_FILES = [
    os.path.join(TESTS_DIR, "data", name) for name in ["inputs1.sdf", "inputs2.sdf"]
]


@pytest.fixture
def space() -> ChemicalSpaceBaseLayer:
    return ChemicalSpaceBaseLayer.from_smi(INPUT_SMI_FILES[0])


@pytest.fixture
def other_space() -> ChemicalSpaceBaseLayer:
    return ChemicalSpaceBaseLayer.from_smi(INPUT_SMI_FILES[1])


@pytest.mark.parametrize("input", INPUT_SMI_FILES[:1] + INPUT_SDF_FILES[:1])
def test_classmethods(input: str) -> None:
    if input.endswith(".smi"):
        space = ChemicalSpaceBaseLayer.from_smi(input)
    else:
        space = ChemicalSpaceBaseLayer.from_sdf(input)

    assert len(space) == 10

    assert len(space.mols) == 10
    assert isinstance(space.mols, tuple)
    assert isinstance(space.mols[0], Mol)

    assert space.indices is not None
    assert len(space.indices) == 10
    assert isinstance(space.indices[0], str)

    assert space.scores is None


def test_save(space: ChemicalSpaceBaseLayer, tmp_path: Path) -> None:
    space.to_sdf(str(tmp_path / "test.sdf"))
    space_loaded = ChemicalSpaceBaseLayer.from_sdf(str(tmp_path / "test.sdf"))

    assert space == space_loaded

    space.to_smi(str(tmp_path / "test.smi"))
    space_loaded = ChemicalSpaceBaseLayer.from_smi(str(tmp_path / "test.smi"))

    assert space == space_loaded


def test_slicing(space: ChemicalSpaceBaseLayer) -> None:
    entry = space[0]
    assert isinstance(entry, tuple)
    assert isinstance(entry[0], Mol)
    assert isinstance(entry[1], str)
    assert entry[2] is None

    entries = space[:2]

    assert len(entries) == 3
    assert len(entries[0]) == len(entries[1]) == 2  # type: ignore
    assert entries[2] is None
    assert isinstance(entries[0], tuple)
    assert isinstance(entries[0][0], Mol)
    assert isinstance(entries[1][0], str)  # type: ignore

    space_slice: ChemicalSpaceBaseLayer = space.slice(start=0, stop=10, step=2)

    assert isinstance(space_slice, ChemicalSpaceBaseLayer)
    assert len(space_slice) == 5

    mask = [True, False] * 5
    space_mask: ChemicalSpaceBaseLayer = space.mask(mask=mask)

    assert isinstance(space_mask, ChemicalSpaceBaseLayer)
    assert len(space_mask) == 5

    assert space_slice == space_mask

    space_chunks: Generator[ChemicalSpaceBaseLayer, None, None] = space.chunks(
        chunk_size=3
    )
    space_chunks_lst = list(space_chunks)
    space_chunks_lst_sizes = [len(chunk) for chunk in space_chunks_lst]

    assert isinstance(space_chunks, GeneratorType)
    assert len(space_chunks_lst) == 4
    assert space_chunks_lst_sizes == [3, 3, 3, 1]
    assert space_chunks_lst[0] == space.slice(start=0, stop=3)


def test_copy(space: ChemicalSpaceBaseLayer) -> None:
    space_shallow: ChemicalSpaceBaseLayer = space.copy(deep=False)
    assert space == space_shallow
    assert id(space.mols[0]) == id(space_shallow.mols[0])

    space_deep: ChemicalSpaceBaseLayer = space.copy(deep=True)
    assert space == space_deep
    assert id(space.mols[0]) != id(space_deep.mols[0])


def test_deduplicate(space: ChemicalSpaceBaseLayer) -> None:
    space_twice = space + space

    space_dedup: ChemicalSpaceBaseLayer = space_twice.deduplicate()
    assert len(space_dedup) == len(space)
    assert space_dedup == space


def test_dual_operations(
    space: ChemicalSpaceBaseLayer, other_space: ChemicalSpaceBaseLayer
) -> None:
    assert space != other_space
    assert space == space
    assert other_space == other_space

    if other_space.indices is None:
        idx = None
    else:
        idx = other_space.indices[0]

    space_add: ChemicalSpaceBaseLayer = space.copy()
    space_add.add(mol=other_space.mols[0], idx=idx)
    assert len(space_add) == len(space) + 1

    combined_spaces = space + other_space

    assert len(combined_spaces) == 10 + 15
    assert combined_spaces.slice(0, 10) == space
    assert combined_spaces.slice(10, None) == other_space

    subtracted_spaces = other_space - space
    assert len(subtracted_spaces) == 10

    empty_space = space - space
    assert len(empty_space) == 0
