from functools import reduce
from typing import Any, Callable, Iterable, List, TypeAlias

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, inchi  # type: ignore

Number: TypeAlias = int | float
MaybeIndex: TypeAlias = None | Any
MaybeScore: TypeAlias = None | Number
MolOrSmiles: TypeAlias = Mol | str
IntOrNone: TypeAlias = int | None
SliceType: TypeAlias = slice


SEED: int = 42


def reduce_sum(*objects: Any) -> Any:
    """
    Returns the sum of all the objects passed as arguments.

    Args:
        *objects: Variable number of objects to be summed.

    Returns:
        The sum of all the objects.

    """
    return reduce(lambda x, y: x + y, objects)


def hash_mol(mol: Mol) -> str:
    """
    Returns the InChIKey hash of a given molecule.

    Args:
        mol (Mol): The molecule to be hashed.

    Returns:
        str: The InChIKey hash of the molecule.
    """
    return inchi.MolToInchiKey(mol)


def ecfp4_featurizer(mol: Mol, radius: int = 2, n_bits: int = 1024) -> List[int]:
    """
    Calculates the ECFP4 fingerprint for a given molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to calculate the fingerprint for.
        radius (int, optional): The radius of the fingerprint. Defaults to 2.
        n_bits (int, optional): The number of bits in the fingerprint. Defaults to 1024.

    Returns:
        List[int]: The ECFP4 fingerprint as a list of integers.
    """
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits).ToList()


def smiles2mol(smiles: str) -> Mol:
    """
    Convert a SMILES string to a molecule object.

    Args:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        Mol: The molecule object generated from the SMILES string.

    Raises:
        ValueError: If the SMILES string cannot be parsed into a molecule object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    return mol


def safe_smiles2mol(smiles_or_mol: MolOrSmiles) -> Mol:
    """
    Convert a SMILES string or RDKit Mol object to an RDKit Mol object.

    Args:
        smiles_or_mol (MolOrSmiles): A RDKit Mol object or a SMILES string.

    Returns:
        Mol: The RDKit Mol object.

    """
    if isinstance(smiles_or_mol, Mol):
        return smiles_or_mol
    return smiles2mol(smiles_or_mol)


def factory(cls, *args, **kwargs) -> Any:
    """
    Create an instance of the given class with the provided arguments.

    Args:
        cls: The class to create an instance of.
        *args: Positional arguments to pass to the class constructor.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the given class.

    """
    return type(cls)(*args, **kwargs)


def parallel_map(
    func: Callable, iterable: Iterable[Any], n_jobs: int = -1
) -> List[Any]:
    """
    Apply a function to each item in the iterable in parallel using multiple processes.

    Args:
        func (Callable): The function to apply to each item.
        iterable (Iterable[Any]): The iterable containing the items to apply the function to.
        n_jobs (int, optional): The number of processes to use for parallel execution.
            Defaults to -1, which uses all available processors.

    Returns:
        List[Any]: A list containing the results of applying the function to each item in the iterable.
    """
    return list(Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in iterable))
