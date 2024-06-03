from .layers.acquisition import ChemicalSpaceAcquisitionLayer
from .layers.clustering import ChemicalSpaceClusteringLayer
from .layers.diversity import ChemicalSpaceDiversityLayer
from .layers.neighbors import ChemicalSpaceNeighborsLayer
from .layers.projection import ChemicalSpaceProjectionLayer


class ChemicalSpace(
    ChemicalSpaceClusteringLayer,
    ChemicalSpaceNeighborsLayer,
    ChemicalSpaceProjectionLayer,
    ChemicalSpaceAcquisitionLayer,
    ChemicalSpaceDiversityLayer,
):
    """
    ChemicalSpace is a class that combines different layers to create a
    complete representation of a chemical space"""

    pass
