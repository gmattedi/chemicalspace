import os
from pathlib import Path
from types import GeneratorType
from typing import Generator

import numpy as np
import pytest
from rdkit.Chem.rdchem import Mol

from chemicalspace.layers.base import ChemicalSpaceBaseLayer
from chemicalspace.utils import (
    MolFeaturizerType,
    ecfp4_featurizer,
    maccs_featurizer,
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SMI_FILES = [
    os.path.join(TESTS_DIR, "data", name) for name in ["inputs1.smi", "inputs2.smi.gz"]
]
INPUT_SDF_FILES = [
    os.path.join(TESTS_DIR, "data", name) for name in ["inputs1.sdf", "inputs2.sdf.gz"]
]


@pytest.fixture
def space() -> ChemicalSpaceBaseLayer:
    return ChemicalSpaceBaseLayer.from_smi(INPUT_SMI_FILES[0])


@pytest.fixture
def other_space() -> ChemicalSpaceBaseLayer:
    return ChemicalSpaceBaseLayer.from_smi(INPUT_SMI_FILES[1])


@pytest.mark.parametrize(
    "input_file,featurizer,metric",
    [
        (INPUT_SMI_FILES[0], ecfp4_featurizer, "jaccard"),
        (INPUT_SMI_FILES[0], maccs_featurizer, "jaccard"),
        (INPUT_SDF_FILES[0], ecfp4_featurizer, "euclidean"),
        (INPUT_SDF_FILES[0], maccs_featurizer, "euclidean"),
    ],
)
def test_classmethods(
    input_file: str, featurizer: MolFeaturizerType, metric: str
) -> None:
    if input_file.endswith(".smi"):
        space = ChemicalSpaceBaseLayer.from_smi(
            input_file, featurizer=featurizer, metric=metric
        )
    else:
        space = ChemicalSpaceBaseLayer.from_sdf(
            input_file, featurizer=featurizer, metric=metric
        )

    assert len(space) == 10

    assert len(space.mols) == 10
    assert isinstance(space.mols, np.ndarray)
    assert isinstance(space.mols[0], Mol)

    assert space.indices is not None
    assert len(space.indices) == 10
    assert isinstance(space.indices[0], str)

    assert space.scores is None


@pytest.mark.parametrize("gzipped", [False, True])
def test_smi_io(space: ChemicalSpaceBaseLayer, tmp_path: Path, gzipped: bool) -> None:
    if gzipped:
        fname = "test.smi.gz"
    else:
        fname = "test.smi"

    space.to_smi(str(tmp_path / fname))

    assert (tmp_path / fname).exists()

    space_loaded = ChemicalSpaceBaseLayer.from_smi(str(tmp_path / fname))

    assert space == space_loaded


@pytest.mark.parametrize("gzipped", [False, True])
def test_sdf_io(space: ChemicalSpaceBaseLayer, tmp_path: Path, gzipped: bool) -> None:
    if gzipped:
        fname = "test.sdf.gz"
    else:
        fname = "test.sdf"

    space.to_sdf(str(tmp_path / fname))

    assert (tmp_path / fname).exists()

    space_loaded = ChemicalSpaceBaseLayer.from_sdf(str(tmp_path / fname))

    assert space == space_loaded


@pytest.mark.parametrize("featurizer", [ecfp4_featurizer, maccs_featurizer])
def test_features(space: ChemicalSpaceBaseLayer, featurizer: MolFeaturizerType) -> None:
    space.featurizer = featurizer

    assert space._features is None
    assert isinstance(space.features, np.ndarray)

    if featurizer == ecfp4_featurizer:
        assert space.features.shape == (10, 1024)
    elif featurizer == maccs_featurizer:
        assert space.features.shape == (10, 167)
    else:
        raise ValueError("Invalid featurizer")

    # Check that the features are cached
    assert space._features is not None


def test_slice_mask_chunk(space: ChemicalSpaceBaseLayer) -> None:
    # Compute the features
    _ = space.features

    entry: ChemicalSpaceBaseLayer = space[0]
    assert isinstance(entry, type(space))
    assert isinstance(entry.mols[0], Mol)
    if entry.indices is not None:
        assert isinstance(entry.indices[0], str)
    assert entry.scores is None

    entries: ChemicalSpaceBaseLayer = space[:3]

    assert isinstance(entries, type(space))
    assert len(entries) == 3
    assert entries.scores is None
    assert isinstance(entries.mols, np.ndarray)
    assert isinstance(entries.mols[0], Mol)
    assert isinstance(entries.indices[0], str)  # type: ignore

    space_slice: ChemicalSpaceBaseLayer = space[0:10:2]

    assert isinstance(space_slice, ChemicalSpaceBaseLayer)
    assert len(space_slice) == 5
    # Check features, but bypass `.features` to check if the array has been assigned when instantiating
    assert np.allclose(space_slice._features, space._features[::2])  # type: ignore

    mask = [True, False] * 5
    space_mask: ChemicalSpaceBaseLayer = space[mask]

    assert isinstance(space_mask, ChemicalSpaceBaseLayer)
    assert len(space_mask) == 5
    assert np.allclose(space_mask._features, space._features[::2])  # type: ignore

    assert space_slice == space_mask

    space_chunks: Generator[ChemicalSpaceBaseLayer, None, None] = space.chunks(
        chunk_size=3
    )
    space_chunks_lst = list(space_chunks)
    space_chunks_lst_sizes = [len(chunk) for chunk in space_chunks_lst]

    assert isinstance(space_chunks, GeneratorType)
    assert len(space_chunks_lst) == 4
    assert space_chunks_lst_sizes == [3, 3, 3, 1]
    assert space_chunks_lst[0] == space[0:3]
    assert np.allclose(space_chunks_lst[0]._features, space._features[:3])  # type: ignore
    assert np.allclose(space_chunks_lst[-1]._features, space._features[-1:])  # type: ignore


def test_attribute_inheritance(space: ChemicalSpaceBaseLayer) -> None:
    """
    Test that the class factory properly propagates attributes to the new class.
    """

    class DerivedSpace(ChemicalSpaceBaseLayer):
        def __init__(self, new_parameter: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.new_parameter = new_parameter

    derived_space = DerivedSpace(
        new_parameter=42, mols=space.mols, indices=space.indices
    )

    derived_space_slice: DerivedSpace = derived_space[:5]
    assert derived_space_slice.new_parameter == 42

    derived_space_mask: DerivedSpace = derived_space[[True, False] * 5]
    assert derived_space_mask.new_parameter == 42

    derived_space_chunks: Generator[DerivedSpace, None, None] = derived_space.chunks(3)
    for chunk in derived_space_chunks:
        assert chunk.new_parameter == 42

    derived_space_copy: DerivedSpace = derived_space.copy()
    assert derived_space_copy.new_parameter == 42

    derived_space_add: DerivedSpace = derived_space + derived_space
    assert derived_space_add.new_parameter == 42

    derived_space_sub: DerivedSpace = derived_space - derived_space
    assert derived_space_sub.new_parameter == 42

    derived_space_dedup: DerivedSpace = derived_space.deduplicate()
    assert derived_space_dedup.new_parameter == 42


def test_copy(space: ChemicalSpaceBaseLayer) -> None:
    space_shallow: ChemicalSpaceBaseLayer = space.copy(deep=False)
    assert space == space_shallow
    assert id(space.mols[0]) == id(space_shallow.mols[0])

    space_deep: ChemicalSpaceBaseLayer = space.copy(deep=True)
    assert space == space_deep
    assert id(space.mols[0]) != id(space_deep.mols[0])


@pytest.mark.parametrize("use_indices", [True, False])
def test_hashing(use_indices: bool, input_file: str = INPUT_SMI_FILES[0]) -> None:
    space = ChemicalSpaceBaseLayer.from_smi(input_file, hash_indices=use_indices)
    space_noindices = ChemicalSpaceBaseLayer.from_smi(input_file, hash_indices=False)

    assert space == space
    if use_indices:
        assert space != space_noindices
    else:
        assert space == space_noindices


def test_deduplicate(space: ChemicalSpaceBaseLayer) -> None:
    space_twice = space + space

    space_dedup: ChemicalSpaceBaseLayer = space_twice.deduplicate()
    assert len(space_dedup) == len(space)
    assert space_dedup == space


def test_dual_operations(
    space: ChemicalSpaceBaseLayer, other_space: ChemicalSpaceBaseLayer
) -> None:
    # Compute the features
    _ = space.features, other_space.features

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
    assert np.allclose(space_add._features, np.vstack([space._features, other_space._features[0]]))  # type: ignore

    combined_spaces = space + other_space

    assert len(combined_spaces) == 10 + 15
    assert combined_spaces[0:10] == space
    assert combined_spaces[10:] == other_space
    assert combined_spaces._features is not None
    assert np.allclose(combined_spaces._features, np.vstack([space._features, other_space._features]))  # type: ignore

    subtracted_spaces = other_space - space
    assert len(subtracted_spaces) == 10
    assert subtracted_spaces._features is not None
    assert np.allclose(subtracted_spaces._features, other_space._features[:-5])  # type: ignore

    empty_space = space - space
    assert len(empty_space) == 0
    assert empty_space._features is not None
    assert empty_space._features.size == 0
