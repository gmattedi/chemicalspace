from .layers.acquisition import ChemicalSpaceAcquisitionLayer
from .layers.clustering import ChemicalSpaceClusteringLayer
from .layers.neigbors import ChemicalSpaceNeighborsLayer
from .layers.projection import ChemicalSpaceProjectionLayer


class ChemicalSpace(
    ChemicalSpaceClusteringLayer,
    ChemicalSpaceNeighborsLayer,
    ChemicalSpaceProjectionLayer,
    ChemicalSpaceAcquisitionLayer,
):
    """
    ChemicalSpace is a class that combines different layers to create a
    complete representation of a chemical space"""

    pass
