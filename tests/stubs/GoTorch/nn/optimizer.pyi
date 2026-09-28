from GoTorch.backend.native_backend import NDArray as NDArray
from GoTorch.nn.module import Module as Module
from GoTorch.tensor import Tensor as Tensor
from _typeshed import Incomplete
from typing import Sequence

class Optimizer:
    """Base class for all neural network optimizers.

    Attributes:
        parameters: List of learnable Tensor parameters to optimize.
        lr: Learning rate.
        b1: First moment decay rate (or momentum factor).
        b2: Second moment decay rate.
    """
    parameters: list[Tensor]
    lr: Incomplete
    b1: Incomplete
    b2: Incomplete
    def __init__(self, module: Module | Sequence[Tensor], lr: float, b1: float, b2: float) -> None:
        """Initializes class variables and registers parameters to optimize.

        Args:
            module: A Module instance or an iterable sequence of Tensor parameters.
            lr: Learning rate.
            b1: First moment decay rate.
            b2: Second moment decay rate.
        """
    def zero_grad(self) -> None:
        """Resets the gradients of all managed parameters to None."""
    def step(self) -> None:
        """Performs a single optimization parameter update step.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """

class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Maintains exponential moving averages of past gradients (first moment)
    and past squared gradients (second moment).

    Attributes:
        parameters: List of learnable Tensor parameters to optimize.
        lr: Learning rate.
        b1: Exponential decay rate for first moment estimates.
        b2: Exponential decay rate for second moment estimates.
        eps: Small constant added to denominator for numerical stability.
        t: Integer step counter for bias corrections.
        m: Dictionary mapping parameter id to first moment estimates.
        v: Dictionary mapping parameter id to second moment estimates.
    """
    eps: Incomplete
    t: int
    m: dict[int, list[float]]
    v: dict[int, list[float]]
    def __init__(self, module: Module | Sequence[Tensor], lr: float, b1: float, b2: float, eps: float = 1e-08) -> None:
        """Initializes the Adam optimizer and sets moments to 0.

        Args:
            module: A Module instance or sequence of Tensors to optimize.
            lr: Learning rate.
            b1: Exponential decay rate for first moment estimates.
            b2: Exponential decay rate for second moment estimates.
            eps: Term added to denominator for numerical stability. Defaults to 1e-8.
        """
    def step(self) -> None:
        """Executes one step of Adam optimization.

        1. Increments global optimization step counter.
        2. Iterates through managed parameters and initializes moment buffers if needed.
        3. Updates biased first and second moment moving averages using current gradients.
        4. Computes bias-corrected moments and calculates parameter delta updates.
        5. Applies updates to parameter data.
        """
