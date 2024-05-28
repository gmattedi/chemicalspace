import warnings
from abc import ABC
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Mol  # type: ignore
from typing_extensions import TypeAlias

from chemicalspace.layers import utils
from .utils import (
    MaybeIndex,
    MaybeScore,
    MolFeaturizerType,
    MolOrSmiles,
    Number,
    SliceType,
    factory,
    parallel_map,
)

T = TypeVar("T", bound="ChemicalSpaceBaseLayer")
ScoreArray: TypeAlias = Union[NDArray[np.int_], NDArray[np.float_], NDArray[np.bool_]]


class ChemicalSpaceBaseLayer(ABC):
    """
    The base class for the ChemicalSpace object.
    Implements the core functionality for the ChemicalSpace object.
    """

    def __init__(
        self,
        mols: Sequence[MolOrSmiles],
        indices: Optional[Sequence[Any]] = None,
        scores: Optional[Sequence[Number]] = None,
        featurizer: MolFeaturizerType = utils.ecfp4_featurizer,
        metric: str = "jaccard",
        features: Optional[NDArray[Any]] = None,
        hash_indices: bool = False,
        n_jobs: int = 1,
    ) -> None:
        """
        Initializes a ChemicalSpace object.

        Args:
            mols (Sequence[Mol | str]): A sequence of RDKit Mol objects or SMILES strings.
            indices (Optional[Sequence[Any]], optional): A sequence of indices for the molecules. Defaults to None.
            scores (Optional[Sequence[Number]], optional): A sequence of scores for the molecules. Defaults to None.
            featurizer (MolFeaturizerType, optional): The featurizer to use for the molecules. Defaults to utils.ecfp4_featurizer.
            metric (str, optional): The sklearn/scipy metric to use for the featurizer. Defaults to "jaccard".
            features (Optional[NDArray[Any]], optional): Precomputed features for the molecules. Defaults to None.
            hash_indices (bool, optional): Whether to include indices in the hash. Defaults to False.
            n_jobs (int, optional): The number of parallel jobs to run. Defaults to 1.


        Raises:
            ValueError: If the number of indices does not match the number of molecules.
            ValueError: If the number of scores does not match the number of molecules.
        """
        mols_m = np.array((parallel_map(utils.safe_smiles2mol, mols, n_jobs=n_jobs)))
        self.mols: NDArray[Mol] = np.array(mols_m)

        self.indices: Union[NDArray[Any], None] = (
            np.array(indices) if indices is not None else None
        )
        self.scores: Union[ScoreArray, None] = (
            np.array(scores) if scores is not None else None
        )
        self.n_jobs = n_jobs
        self.featurizer = featurizer
        self.metric = metric
        self.hash_indices = hash_indices

        if self.indices is not None and len(self.indices) != len(self.mols):
            raise ValueError("Number of indices must match number of molecules")
        if self.scores is not None and len(self.scores) != len(self.mols):
            raise ValueError("Number of scores must match number of molecules")
        if (features is not None) and (len(features) != len(self.mols)):
            raise ValueError("Number of features must match number of molecules")

        self._features: Union[NDArray[Any], None] = features

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
        mol_m = mol if isinstance(mol, Mol) else utils.smiles2mol(mol)
        self.mols = np.append(self.mols, mol_m)

        if self.indices is not None:
            if idx is None:
                raise ValueError("Indices must be provided for all molecules")
            else:
                self.indices = np.append(self.indices, idx)

        if self.scores is not None:
            if score is None:
                raise ValueError("Scores must be provided for all molecules")
            else:
                self.scores = np.append(self.scores, score)

        if self._features is not None:
            feats = np.array(self.featurizer(mol_m)).reshape(1, -1)
            self._features = np.r_[self._features, feats]

    def chunks(self, chunk_size: int) -> Generator[T, None, None]:  # type: ignore
        """
        Split the ChemicalSpaceBaseLayer into chunks of a given size.

        Args:
            chunk_size (int): The size of each chunk.

        Yields:
            Generator[ChemicalSpaceBaseLayer, None, None]: A generator of ChemicalSpaceBaseLayer objects.
        """
        for i in range(0, len(self), chunk_size):
            yield self[i : i + chunk_size]

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

        mols_hashes = parallel_map(utils.hash_mol, self.mols, n_jobs=self.n_jobs)

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

        idx: Union[List[Any], None] = idx_lst if self.indices is not None else None
        scores: Union[List[Number], None] = (
            scores_lst if self.scores is not None else None
        )

        if self._features is not None:
            features = self._features[features_idx]
        else:
            features = None

        return factory(
            self,
            mols=mols_lst,
            indices=idx,
            scores=scores,
            features=features,
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
                parallel_map(self.featurizer, self.mols, n_jobs=self.n_jobs), dtype=int
            )

        return self._features

    @classmethod
    def from_smi(cls: Type[T], path: str, **kwargs) -> T:
        """
        Create a ChemicalSpaceBaseLayer object from a file containing SMILES strings.

        Args:
            path (str): The path to the file containing SMILES strings.
                Can be gzipped
            kwargs (Any): Additional keyword arguments to pass to the constructor.

        Returns:
            ChemicalSpaceBaseLayer: A ChemicalSpaceBaseLayer object created from the SMILES strings.

        """

        supplier = utils.smi_supplier(path)

        mols_lst: List[Mol] = []
        indices_lst: List[str] = []

        for mol in supplier:  # type: ignore
            if mol is None:
                warnings.warn("Failed to parse molecule")
                continue

            mols_lst.append(mol)
            indices_lst.append(str(mol.GetProp("_Name")))

        return cls(
            mols=mols_lst,
            indices=indices_lst,
            scores=None,
            **kwargs,
        )

    @classmethod
    def from_sdf(
        cls: Type[T], path: str, scores_prop: Optional[str] = None, **kwargs
    ) -> T:
        """
        Create a ChemicalSpaceBaseLayer object from an SDF file.

        Args:
            path (str): The path to the SDF file. Can be gzipped.
            scores_prop (Optional[str]): The property name in the SDF file that contains the scores. Default is None.
            kwargs (Any): Additional keyword arguments to pass to the constructor.

        Returns:
            ChemicalSpaceBaseLayer: The ChemicalSpaceBaseLayer object created from the SDF file.

        """
        supplier = utils.sdf_supplier(path)

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
                mols=mols_lst,
                indices=indices_lst,
                scores=scores_lst,
            )
        else:
            return cls(
                mols=mols_lst,
                indices=indices_lst,
                scores=None,
                **kwargs,
            )

    def to_smi(self, path: str) -> None:
        """
        Write the molecules in the chemical space to a file as SMILES strings.

        Args:
            path (str): The path to the output file. Can be gzipped

        Returns:
            None
        """
        if self.indices is None:
            indices = [""] * len(self)
        else:
            indices = list(self.indices)

        utils.smi_writer(path, self.mols, indices)

    def to_sdf(self, path: str, scores_prop: Optional[str] = None) -> None:
        """
        Write the molecules in the chemical space to an SDF file.

        Args:
            path (str): The path to the output file. Can be gzipped.
            scores_prop (Optional[str]): The property name to use for the scores. Default is None.

        Returns:
            None
        """

        if self.indices is None:
            indices = [""] * len(self)
        else:
            indices = list(map(str, self.indices))

        if scores_prop is not None:
            if self.scores is None:
                raise ValueError("Scores are not available")
            else:
                for mol, idx, score in zip(self.mols, indices, self.scores):
                    mol.SetProp("_Name", idx)
                    mol.SetProp(scores_prop, str(score))
        else:
            for mol, idx in zip(self.mols, indices):
                mol.SetProp("_Name", idx)

        utils.sdf_writer(path, self.mols)

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

        mols = np.concatenate([self.mols, other.mols])
        if (self.indices is None) or (other.indices is None):
            if (self.indices is None) != (other.indices is None):
                warnings.warn(
                    "Both spaces should have indices to concatenate. Indices will be None"
                )
            idx = None
        else:
            idx = np.concatenate([self.indices, other.indices])

        if (self.scores is None) or (other.scores is None):
            if (self.scores is None) != (other.scores is None):
                warnings.warn(
                    "One or more spaces do not have scores. Scores will be None"
                )
            score = None
        else:
            score = np.concatenate([self.scores, other.scores])

        if (self._features is None) or (other._features is None):
            # if (self._features is None) != (other._features is None):
            #     warnings.warn(
            #         "One or more spaces do not have features. Features will be None"
            #     )
            features = None
        else:
            features = np.vstack([self._features, other._features])

        return factory(
            self,
            mols=mols,
            indices=idx,
            scores=score,
            features=features,
        )

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

        cache = set(parallel_map(utils.hash_mol, other.mols, n_jobs=self.n_jobs))

        mols_lst: List[Mol] = []
        indices_lst: List[Any] = []
        scores_lst: List[Number] = []
        features_idx: List[int] = []

        mols_hashes = parallel_map(utils.hash_mol, self.mols, n_jobs=self.n_jobs)

        for i in range(len(self)):
            mol: Mol = self.mols[i]
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

        idx: Union[List[Any], None] = indices_lst if self.indices is not None else None
        scores: Union[List[Number], None] = (
            scores_lst if self.scores is not None else None
        )
        features = self._features[features_idx] if self._features is not None else None

        return factory(
            self,
            mols=mols_lst,
            indices=idx,
            scores=scores,
            features=features,
            featurizer=self.featurizer,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def __getitem__(
        self, idx: Union[int, SliceType, NDArray[np.bool_], List[bool], List[int]]
    ) -> T:  # type: ignore
        """
        Retrieve the item(s) at the specified index or slice.

        Args:
            idx: The index, slice, mask, or list of indices to retrieve.

        Returns:
            ChemicalSpaceBaseLayer: A new ChemicalSpaceBaseLayer object containing the item(s) at the specified index or slice.
        """

        if isinstance(idx, int):
            idx = [idx]  # type: ignore

        mol = self.mols[idx]
        mol_idx = self.indices[idx] if self.indices is not None else None
        score = self.scores[idx] if self.scores is not None else None
        features = self._features[idx] if self._features is not None else None

        return factory(
            self,
            mols=mol,
            indices=mol_idx,
            scores=score,
            features=features,
        )

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

        Args:
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
        mols = [Mol(mol) for mol in self.mols]
        indices = [idx for idx in self.indices] if self.indices is not None else None
        scores = [score for score in self.scores] if self.scores is not None else None
        features = self._features.copy() if self._features is not None else None
        return factory(
            self,
            mols=mols,
            indices=indices,
            scores=scores,
            features=features,
        )

    @staticmethod
    def hash_mol(mol: Mol) -> str:
        """
        Compute the hash of a molecule.

        Args:
            mol (Mol): The molecule to hash.

        Returns:
            str: The hash of the molecule.
        """
        return Chem.MolToInchiKey(mol)

    def __hash__(self) -> int:
        inchi_keys: List[str] = parallel_map(
            self.hash_mol, self.mols, n_jobs=self.n_jobs
        )

        if self.hash_indices:
            indices = self.indices if self.indices is not None else [""] * len(self)
            mol_strings = [
                f"{inchi_key}@{idx}" for inchi_key, idx in zip(inchi_keys, indices)
            ]
        else:
            mol_strings = inchi_keys

        return hash(frozenset(mol_strings))

    def __repr__(self) -> str:
        idx_repr = len(self.indices) if self.indices is not None else "No"
        scores_repr = len(self.scores) if self.scores is not None else "No"
        return f"<{self.name}: {len(self)} molecules | {idx_repr} indices | {scores_repr} scores>"

    def __str__(self) -> str:
        return self.__repr__()
