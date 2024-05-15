from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from chemicalspace.layers.base import ChemicalSpaceBaseLayer, T


class BaseAcquisitionStrategy(ABC):
    def __init__(
        self,
        inputs: Sequence[Any] | NDArray[Any],
        scores: Optional[Sequence[Any] | NDArray[Any]] = None,
    ):
        self.inputs = np.array(inputs)
        if scores is None:
            scores = np.zeros(len(inputs))
        self.scores = np.array(scores)

    @abstractmethod
    def __call__(self, n: int) -> List[int]:
        raise NotImplementedError


class BaseAcquisitionStrategyRequiresScores(BaseAcquisitionStrategy):
    def __init__(
        self,
        inputs: Sequence[Any] | NDArray[Any],
        scores: Sequence[Any] | NDArray[Any],
    ):
        super().__init__(inputs, scores)
        if scores is None:
            raise ValueError("Scores are required for this acquisition strategy.")


class RandomStrategy(BaseAcquisitionStrategy):
    def __call__(self, n: int) -> List[int]:
        idx = np.random.choice(len(self.inputs), n, replace=False)
        return idx.tolist()


class GreedyStrategy(BaseAcquisitionStrategyRequiresScores):
    def __call__(self, n: int) -> List[int]:
        idx = np.argsort(self.scores)[-n:]
        return idx.tolist()


STRATEGIES: TypeAlias = Literal["random", "greedy"]
strategies_dict = {"random": RandomStrategy, "greedy": GreedyStrategy}


def pick_samples(
    n: int,
    strategy: STRATEGIES,
    inputs: Sequence[Any] | NDArray[Any],
    scores: Optional[Sequence[Any] | NDArray[Any]] = None,
) -> NDArray[np.int_]:
    strategy_class = strategies_dict[strategy]
    strategy_instance = strategy_class(inputs, scores)

    return np.array(strategy_instance(n), dtype=int)


class ChemicalSpaceAcquisitionLayer(ChemicalSpaceBaseLayer):
    def pick(self, n: int, strategy: STRATEGIES = "random") -> T:  # type: ignore
        pick_idx = pick_samples(
            n=n, strategy=strategy, inputs=self.features, scores=self.scores
        )
        mask = np.zeros(len(self.features), dtype=bool)
        mask[pick_idx] = True

        return self.mask(mask)
