import warnings
from abc import ABC
from functools import cached_property
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Mol  # type: ignore

from .utils import (
    IntOrNone,
    MaybeIndex,
    MaybeScore,
    MolOrSmiles,
    Number,
    SliceType,
    ecfp4_featurizer,
    factory,
    hash_mol,
    parallel_map,
    safe_smiles2mol,
    smiles2mol,
)

T = TypeVar("T", bound="ChemicalSpaceBaseLayer")


class ChemicalSpaceBaseLayer(ABC):
    def __init__(
        self,
        mols: Tuple[MolOrSmiles, ...],
        indices: Optional[Tuple[Any, ...]] = None,
        scores: Optional[Tuple[Number, ...]] = None,
        features: Optional[NDArray[np.int_]] = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Initializes a ChemicalSpace object.

        Args:
            mols (Tuple[MolOrSmiles, ...]): A tuple of molecules or SMILES strings.
            indices (Optional[Tuple[Any, ...]], optional): A tuple of indices. Defaults to None.
            scores (Optional[Tuple[Number, ...]], optional): A tuple of scores. Defaults to None.
            features (Optional[NDArray[np.int_]], optional): A numpy array of Morgan fingeprints. Defaults to None.
            n_jobs (int, optional): The number of jobs to use for parallel processing. Defaults to 1.

        Raises:
            ValueError: If the number of indices does not match the number of molecules.
            ValueError: If the number of scores does not match the number of molecules.
        """
        mols_m = tuple(parallel_map(safe_smiles2mol, mols, n_jobs=n_jobs))
        self.mols: Tuple[Mol, ...] = mols_m
        self.indices = indices
        self.scores = scores
        self.n_jobs = n_jobs

        if self.indices is not None and len(self.indices) != len(self.mols):
            raise ValueError("Number of indices must match number of molecules")
        if self.scores is not None and len(self.scores) != len(self.mols):
            raise ValueError("Number of scores must match number of molecules")

        self._features: Union[NDArray[np.int_], None] = features

        self.name = self.__class__.__name__

    def add(
        self, mol: MolOrSmiles, idx: MaybeIndex = None, score: MaybeScore = None
    ) -> None:
        """
        Adds a molecule to the chemical space.

        Args:
            mol (MolOrSmiles): The molecule to be added. It can be either a Mol object or a SMILES string.
            idx (MaybeIndex, optional): The index of the molecule. Defaults to None.
            score (MaybeScore, optional): The score associated with the molecule. Defaults to None.

        Raises:
            ValueError: If scores are enabled and score is not provided.

        Returns:
            None
        """
        mol_m = mol if isinstance(mol, Mol) else smiles2mol(mol)
        self.mols += (mol_m,)

        if self.indices is not None:
            self.indices += (idx,)
        if self.scores is not None:
            if score is None:
                raise ValueError("Scores must be provided for all molecules")
            else:
                self.scores += (score,)

        if self._features is not None:
            self._features = np.vstack([self._features, ecfp4_featurizer(mol_m)])

    def chunks(self, chunk_size: int) -> Generator[T, None, None]:  # type: ignore
        """
        Split the ChemicalSpaceBaseLayer into chunks of a given size.

        Args:
            chunk_size (int): The size of each chunk.

        Yields:
            Generator[ChemicalSpaceBaseLayer, None, None]: A generator of ChemicalSpaceBaseLayer objects.
        """
        for i in range(0, len(self), chunk_size):
            yield factory(
                self,
                mols=self.mols[i : i + chunk_size],
                indices=(
                    self.indices[i : i + chunk_size]
                    if self.indices is not None
                    else None
                ),
                scores=(
                    self.scores[i : i + chunk_size] if self.scores is not None else None
                ),
                features=(
                    self._features[i : i + chunk_size]
                    if self._features is not None
                    else None
                ),
            )

    def slice(
        self, start: IntOrNone, stop: IntOrNone, step: IntOrNone = None
    ) -> T:  # type: ignore
        """
        Slice the ChemicalSpaceBaseLayer object based on the given start, stop, and step Args.

        Args:
            start (int or None): The start index of the slice.
            stop (int or None): The stop index of the slice.
            step (int or None): The step size of the slice.

        Returns:
            ChemicalSpaceBaseLayer: A new ChemicalSpaceBaseLayer object containing the sliced data.

        """
        s = slice(start, stop, step)

        return factory(
            self,
            mols=self.mols[s],
            indices=self.indices[s] if self.indices is not None else None,
            scores=self.scores[s] if self.scores is not None else None,
            features=self._features[s] if self._features is not None else None,
        )

    def mask(self, mask: Union[NDArray[np.bool_], List[bool]]) -> T:  # type: ignore
        """
        Applies a boolean mask to the ChemicalSpaceBaseLayer object.

        Args:
            mask: A boolean mask indicating which elements to keep.

        Returns:
            A new ChemicalSpaceBaseLayer object with the masked elements.

        """
        mask_arr = np.array(mask, dtype=bool)

        mols = [mol for mol, mask in zip(self.mols, mask_arr) if mask]
        indices = (
            [idx for idx, mask in zip(self.indices, mask_arr) if mask]
            if self.indices is not None
            else None
        )
        scores = (
            [score for score, mask in zip(self.scores, mask_arr) if mask]
            if self.scores is not None
            else None
        )
        features = self._features[mask_arr] if self._features is not None else None

        return factory(
            self,
            mols=tuple(mols),
            indices=tuple(indices) if indices is not None else None,
            scores=tuple(scores) if scores is not None else None,
            features=features,
        )

    def deduplicate(self) -> T:  # type: ignore
        """
        Remove duplicate molecules from the ChemicalSpaceBaseLayer by
        keeping only the *first* occurrence of each molecule, based on their InChIKey.

        Returns:
            A new ChemicalSpaceBaseLayer object with duplicate molecules removed.
        """
        cache = set()

        mols_lst: List[Mol] = []
        idx_lst: List[Any] = []
        scores_lst: List[Number] = []
        features_idx: List[int] = []

        mols_hashes = parallel_map(hash_mol, self.mols, n_jobs=self.n_jobs)

        for i in range(len(self)):
            mol = self.mols[i]
            mol_hash = mols_hashes[i]

            if mol_hash in cache:
                continue
            else:
                cache.add(mol_hash)
                mols_lst.append(mol)
                if self.indices is not None:
                    idx_lst.append(self.indices[i])

                if self.scores is not None:
                    scores_lst.append(self.scores[i])

                if self._features is not None:
                    features_idx.append(i)

        idx: Optional[Tuple[Any, ...]] = (
            tuple(idx_lst) if self.indices is not None else None
        )
        scores: Optional[Tuple[Number, ...]] = (
            tuple(scores_lst) if self.scores is not None else None
        )

        features = self._features[features_idx] if self._features is not None else None

        return factory(
            self, mols=tuple(mols_lst), indices=idx, scores=scores, features=features
        )

    @cached_property
    def features(self) -> NDArray[np.int_]:
        """
        Calculate the features for each molecule in the chemical space.

        Returns:
            NDArray[np.int_]: An array of features for each molecule.
        """
        if self._features is None:
            self._features = np.array(
                parallel_map(ecfp4_featurizer, self.mols, n_jobs=self.n_jobs), dtype=int
            )

        return self._features

    @classmethod
    def from_smi(cls: Type[T], path: str) -> T:
        """
        Create a ChemicalSpaceBaseLayer object from a file containing SMILES strings.

        Args:
            path (str): The path to the file containing SMILES strings.

        Returns:
            ChemicalSpaceBaseLayer: A ChemicalSpaceBaseLayer object created from the SMILES strings.

        """
        supplier = Chem.SmilesMolSupplier(path, titleLine=False)

        mols_lst: List[Mol] = []
        indices_lst: List[str] = []

        for mol in supplier:  # type: ignore
            if mol is None:
                warnings.warn("Failed to parse molecule")
                continue

            mols_lst.append(mol)
            indices_lst.append(str(mol.GetProp("_Name")))

        return cls(mols=tuple(mols_lst), indices=tuple(indices_lst), scores=None)

    @classmethod
    def from_sdf(cls: Type[T], path: str, scores_prop: Optional[str] = None) -> T:
        """
        Create a ChemicalSpaceBaseLayer object from an SDF file.

        Args:
            path (str): The path to the SDF file.
            scores_prop (Optional[str]): The property name in the SDF file that contains the scores. Default is None.

        Returns:
            ChemicalSpaceBaseLayer: The ChemicalSpaceBaseLayer object created from the SDF file.

        """
        supplier = Chem.SDMolSupplier(path)

        mols_lst: List[Mol] = []
        indices_lst: List[str] = []
        scores_lst: List[float] = []

        for mol in supplier:
            if mol is None:
                warnings.warn("Failed to parse molecule")
                continue

            mols_lst.append(mol)
            indices_lst.append(mol.GetProp("_Name"))

            if scores_prop is not None:
                scores_lst.append(float(mol.GetProp(scores_prop)))

        if scores_prop is not None:
            return cls(
                mols=tuple(mols_lst),
                indices=tuple(indices_lst),
                scores=tuple(scores_lst),
            )
        else:
            return cls(mols=tuple(mols_lst), indices=tuple(indices_lst), scores=None)

    def to_smi(self, path: str) -> None:
        """
        Write the molecules in the chemical space to a file as SMILES strings.

        Args:
            path (str): The path to the output file.

        Returns:
            None
        """
        if self.indices is None:
            indices = [""] * len(self)
        else:
            indices = list(self.indices)

        with open(path, "w") as f:
            for mol, idx in zip(self.mols, indices):
                smi = Chem.MolToSmiles(mol)
                f.write(f"{smi} {idx}\n")

    def to_sdf(self, path: str, scores_prop: Optional[str] = None) -> None:
        """
        Write the molecules in the chemical space to an SDF file.

        Args:
            path (str): The path to the output file.
            scores_prop (Optional[str]): The property name to use for the scores. Default is None.

        Returns:
            None
        """

        if self.indices is None:
            indices = [""] * len(self)
        else:
            indices = list(self.indices)

        if self.scores is None:
            if scores_prop is None:
                scores = [0.0] * len(self)
            else:
                raise ValueError("Scores must be provided to write to SDF")
        else:
            scores = list(self.scores)

        w = Chem.SDWriter(path)

        for mol, idx, score in zip(self.mols, indices, scores):
            mol.SetProp("_Name", idx)
            if scores_prop is not None:
                mol.SetProp(scores_prop, str(score))
            w.write(mol)

        w.close()

    def __len__(self) -> int:
        """
        Returns the number of molecules in the chemical space.

        Returns:
            int: The number of molecules in the chemical space.
        """
        return len(self.mols)

    def __add__(self, other: T) -> T:
        """
        Adds two ChemicalSpaceBaseLayer objects together.

        Args:
            other (ChemicalSpaceBaseLayer): The other ChemicalSpaceBaseLayer object to add.

        Returns:
            ChemicalSpaceBaseLayer: A new ChemicalSpaceBaseLayer object that is the result of the addition.

        Raises:
            TypeError: If the other object is not an instance of ChemicalSpaceBaseLayer.
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"Can only add {self.name} objects")

        mols = self.mols + other.mols
        if (self.indices is None) or (other.indices is None):
            if (self.indices is None) != (other.indices is None):
                warnings.warn(
                    "Both spaces should have indices to concatenate. Indices will be None"
                )
            idx = None
        else:
            idx = self.indices + other.indices

        if (self.scores is None) or (other.scores is None):
            if (self.scores is None) != (other.scores is None):
                warnings.warn(
                    "One or more spaces do not have scores. Scores will be None"
                )
            score = None
        else:
            score = self.scores + other.scores

        if (self._features is None) or (other._features is None):
            if (self._features is None) != (other._features is None):
                warnings.warn(
                    "One or more spaces do not have features. Features will be None"
                )
            features = None
        else:
            features = np.vstack([self._features, other._features])

        return factory(self, mols=mols, indices=idx, scores=score, features=features)

    def __sub__(self, other: T) -> T:
        """
        Subtract another ChemicalSpaceBaseLayer object from the current object.

        Args:
            other (ChemicalSpaceBaseLayer): The ChemicalSpaceBaseLayer object to subtract.

        Returns:
            ChemicalSpaceBaseLayer: A new ChemicalSpaceBaseLayer object that contains the molecules
            from the current object that are not present in the other object.

        Raises:
            TypeError: If the other object is not an instance of ChemicalSpaceBaseLayer.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Can only subtract ChemicalSpace objects")

        cache = set(parallel_map(hash_mol, other.mols, n_jobs=self.n_jobs))

        mols_lst: List[Mol] = []
        indices_lst: List[Any] = []
        scores_lst: List[Number] = []
        features_idx: List[int] = []

        mols_hashes = parallel_map(hash_mol, self.mols, n_jobs=self.n_jobs)

        for i in range(len(self)):
            mol = self.mols[i]
            mol_hash = mols_hashes[i]

            if mol_hash in cache:
                continue
            else:
                mols_lst.append(mol)
                if self.indices is not None:
                    indices_lst.append(self.indices[i])
                if self.scores is not None:
                    scores_lst.append(self.scores[i])
                if self._features is not None:
                    features_idx.append(i)

        idx: Optional[Tuple[Any, ...]] = (
            tuple(indices_lst) if self.indices is not None else None
        )
        scores: Optional[Tuple[Number, ...]] = (
            tuple(scores_lst) if self.scores is not None else None
        )
        features = self._features[features_idx] if self._features is not None else None

        return factory(
            self, mols=tuple(mols_lst), indices=idx, scores=scores, features=features
        )

    def __getitem__(self, idx: int | SliceType) -> Tuple[
        Mol | Tuple[Mol, ...],
        MaybeIndex | Tuple[MaybeIndex, ...],
        MaybeScore | Tuple[MaybeScore, ...],
    ]:
        """
        Retrieve the item(s) at the specified index or slice.

        Args:
            idx (int | SliceType): The index or slice to retrieve the item(s) from.

        Returns:
            Tuple[
                Mol | Tuple[Mol, ...],
                MaybeIndex | Tuple[MaybeIndex, ...],
                MaybeScore | Tuple[MaybeScore, ...],
            ]: A tuple containing the molecule(s), index(es), and score(s) at the specified index or slice.
        """
        mol = self.mols[idx]
        mol_idx = self.indices[idx] if self.indices is not None else None
        score = self.scores[idx] if self.scores is not None else None

        return mol, mol_idx, score

    def __eq__(self, other: object) -> bool:
        """
        Check if two ChemicalSpaceBaseLayer objects are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, type(self)):
            return False

        if len(self) != len(other):
            return False

        return hash(self) == hash(other)

    def copy(self, deep: bool = False) -> T:  # type: ignore
        """
        Create a copy of the object.

        Parameters:
            deep (bool): If True, perform a deep copy of the object. If False, perform a shallow copy.

        Returns:
            T: A copy of the object.
        """
        if not deep:
            return self.__copy__()
        else:
            return self.__deepcopy__({})

    def __copy__(self) -> T:  # type: ignore
        return factory(
            self,
            mols=self.mols,
            indices=self.indices,
            scores=self.scores,
            features=self._features,
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> T:  # type: ignore
        _ = memo
        mols = tuple([Chem.Mol(mol) for mol in self.mols])
        indices = (
            tuple([idx for idx in self.indices]) if self.indices is not None else None
        )
        scores = (
            tuple([score for score in self.scores]) if self.scores is not None else None
        )
        features = self._features.copy() if self._features is not None else None
        return factory(
            self, mols=mols, indices=indices, scores=scores, features=features
        )

    def __hash__(self) -> int:
        return hash(frozenset(parallel_map(hash_mol, self.mols, n_jobs=self.n_jobs)))

    def __repr__(self) -> str:
        idx_repr = len(self.indices) if self.indices is not None else "No"
        scores_repr = len(self.scores) if self.scores is not None else "No"
        return f"<{self.name}: {len(self)} molecules | {idx_repr} indices | {scores_repr} scores>"

    def __str__(self) -> str:
        return self.__repr__()
