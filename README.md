# ChemicalSpace
An Object-Oriented Representation for Chemical Spaces

`ChemicalSpace` is a Python package that provides an object-oriented representation for chemical spaces. It is designed to be used in conjunction with the `RDKit` package, which provides the underlying cheminformatics functionality.

## Installation
To install `ChemicalSpace`, you can use `pip`:

```bash
pip install .
```

## Usage
The main class in `ChemicalSpace` is `ChemicalSpace`. This class is designed to represent a chemical space, which is a collection of molecules. The `ChemicalSpace` class provides a number of methods for working with chemical spaces, including methods for reading and writing chemical spaces, filtering, clustering and picking from chemical spaces.

### Initialization
A `ChemicalSpace` can be initialized from a set of SMILES strings or `RDKit` molecules. It optionally takes molecule indices and scores as arguments.

```python
from chemicalspace import ChemicalSpace

smiles = ('CCO', 'CCN', 'CCl')
indices = ("mol1", "mol2", "mol3")
scores = (0.1, 0.2, 0.3)

space = ChemicalSpace(mols=smiles, indices=indices, scores=scores)

print(space)
```
```text
<ChemicalSpace: 3 molecules | 3 indices | 3 scores>
```

### Reading and Writing
A `ChemicalSpace` can be read from and written to SMI and SDF files.

```python
from chemicalspace import ChemicalSpace

space = ChemicalSpace.from_smi("tests/data/inputs1.smi")
space.to_smi("outputs1.smi")

space = ChemicalSpace.from_sdf("tests/data/inputs1.sdf")
space.to_sdf("outputs1.sdf")
```