from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from chemicalspace.layers.base import ChemicalSpaceBaseLayer, T


class BaseAcquisitionStrategy(ABC):
    """
    BaseAcquisitionStrategy is an abstract class that defines the interface
    for acquisition strategies. Acquisition strategies are classes that
    implement a method to pick samples from a set of inputs.
    """

    def __init__(
        self,
        inputs: Sequence[Any] | NDArray[Any],
        scores: Optional[Sequence[Any] | NDArray[Any]] = None,
    ):
        """
        Initialize the acquisition strategy with input object features and scores

        Args:
            inputs (Sequence[Any] | NDArray[Any]): Inputs features to the acquisition strategy.
            scores (Optional[Sequence[Any] | NDArray[Any]], optional): Scores associated with the inputs.
                Defaults to None.
        """
        self.inputs = np.array(inputs)
        if scores is None:
            scores = np.zeros(len(inputs))
        self.scores = np.array(scores)

    @abstractmethod
    def __call__(self, n: int) -> List[int]:
        """
        Pick n samples from the inputs.

        Args:
            n (int): Number of samples to pick.

        Returns:
            List[int]: List of indices of the picked samples.

        """
        raise NotImplementedError


class BaseAcquisitionStrategyRequiresScores(BaseAcquisitionStrategy, ABC):
    """
    BaseAcquisitionStrategyRequiresScores is an abstract class that defines the
    interface for acquisition strategies that require scores to pick samples.
    """

    def __init__(
        self,
        inputs: Sequence[Any] | NDArray[Any],
        scores: Sequence[Any] | NDArray[Any],
    ):
        """
        Initialize the acquisition strategy with input object features and scores

        Args:
            inputs (Sequence[Any] | NDArray[Any]): Inputs features to the acquisition strategy.
            scores (Optional[Sequence[Any] | NDArray[Any]], optional): Scores associated with the inputs.
                Defaults to None.
        """
        super().__init__(inputs, scores)
        if scores is None:
            raise ValueError("Scores are required for this acquisition strategy.")


class RandomStrategy(BaseAcquisitionStrategy):
    """
    RandomStrategy is an acquisition strategy that picks samples randomly.
    """

    def __call__(self, n: int) -> List[int]:
        """
        Pick n samples from the inputs randomly.

        Args:
            n (int): Number of samples to pick.

        Returns:
            List[int]: List of indices of the picked samples.

        """
        idx = np.random.choice(len(self.inputs), n, replace=False)
        return idx.tolist()


class GreedyStrategy(BaseAcquisitionStrategyRequiresScores):
    """
    GreedyStrategy is an acquisition strategy that picks samples greedily based on scores.
    """

    def __call__(self, n: int) -> List[int]:
        """
        Pick the top n samples from the inputs based on scores.

        Args:
            n (int): Number of samples to pick.

        Returns:
            List[int]: List of indices of the picked samples.

        """
        idx = np.argsort(self.scores)[-n:]
        return idx.tolist()


STRATEGIES: TypeAlias = Literal["random", "greedy"]
strategies_dict = {"random": RandomStrategy, "greedy": GreedyStrategy}


def pick_samples(
    n: int,
    strategy: STRATEGIES,
    inputs: Sequence[Any] | NDArray[Any],
    scores: Optional[Sequence[Any] | NDArray[Any]] = None,
    **strategy_kwargs,
) -> NDArray[np.int_]:
    """
    Pick n samples from the inputs using the specified strategy.

    Args:
        n (int): Number of samples to pick.
        strategy (STRATEGIES): Strategy to use for picking samples.
        inputs (Sequence[Any] | NDArray[Any]): Inputs features to the acquisition strategy.
        scores (Optional[Sequence[Any] | NDArray[Any]], optional): Scores associated with the inputs.
        **strategy_kwargs: Additional keyword arguments to pass to the acquisition strategy.

    Returns:
        NDArray[np.int_]: Array of indices of the picked samples.

    """
    if strategy not in strategies_dict:
        raise ValueError(
            f"Unknown strategy: {strategy}. Allowed strategies: {list(strategies_dict.keys())}"
        )

    strategy_class = strategies_dict[strategy]
    strategy_instance = strategy_class(inputs=inputs, scores=scores, **strategy_kwargs)

    return np.array(strategy_instance(n), dtype=int)


class ChemicalSpaceAcquisitionLayer(ChemicalSpaceBaseLayer):
    """
    ChemicalSpaceAcquisitionLayer is a class that provides acquisition
    functionalities to a chemical space. It is a mixin class that can be
    combined with other layers to create a complete representation of a
    chemical space.
    """

    def pick(self, n: int, strategy: STRATEGIES = "random", **strategy_kwargs) -> T:  # type: ignore
        """
        Pick n samples from the chemical space using the specified strategy.

        Args:
            n (int): Number of samples to pick.
            strategy (STRATEGIES, optional): Strategy to use for picking samples. Defaults to "random".
            **strategy_kwargs: Additional keyword arguments to pass to the acquisition strategy.

        Returns:
            T: A ChemicalSpace object with the picked samples.

        """
        pick_idx = pick_samples(
            n=n,
            strategy=strategy,
            inputs=self.features,
            scores=self.scores,
            **strategy_kwargs,
        )
        mask = np.zeros(len(self.features), dtype=bool)
        mask[pick_idx] = True

        return self.mask(mask)
