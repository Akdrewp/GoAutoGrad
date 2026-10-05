from GoTorch.nn.module import Module as Module
from GoTorch.tensor import Tensor as Tensor
from _typeshed import Incomplete

class MSELoss(Module):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input and target.

    Attributes:
        reduction: Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
    """
    reduction: Incomplete
    def __init__(self, reduction: str = 'mean') -> None:
        """Initializes the MSELoss criterion.

        Args:
            reduction: Specifies the reduction to apply: 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the mean squared error loss between input and target.

        Args:
            input: Predicted values Tensor.
            target: Ground truth target values Tensor.

        Returns:
            Output loss Tensor.
        """

class BCELoss(Module):
    """Creates a criterion that measures the Binary Cross Entropy between the target and
    the input probabilities.

    Attributes:
        reduction: Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
    """
    reduction: Incomplete
    def __init__(self, reduction: str = 'mean') -> None:
        """Initializes the BCELoss criterion.

        Args:
            reduction: Specifies the reduction to apply: 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes binary cross entropy loss between input and target.

        Args:
            input: Predicted probabilities Tensor.
            target: Ground truth binary target Tensor.

        Returns:
            Output loss Tensor.
        """
