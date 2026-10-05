"""Loss functions for neural network training."""

from __future__ import annotations

from GoTorch.autograd.operations import (
    BCELoss as BCELossOp,
)
from GoTorch.autograd.operations import (
    MSELoss as MSELossOp,
)
from GoTorch.nn.module import Module
from GoTorch.tensor import Tensor


class MSELoss(Module):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input and target.

    Attributes:
        reduction: Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initializes the MSELoss criterion.

        Args:
            reduction: Specifies the reduction to apply: 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the mean squared error loss between input and target.

        Args:
            input: Predicted values Tensor.
            target: Ground truth target values Tensor.

        Returns:
            Output loss Tensor.
        """
        out_data = MSELossOp.compute(input.data, target.data, reduction=self.reduction)
        node = Tensor(data=out_data, op=MSELossOp, inputs=[input, target])
        node.reduction = self.reduction
        return node


class BCELoss(Module):
    """Creates a criterion that measures the Binary Cross Entropy between the target and
    the input probabilities.

    Attributes:
        reduction: Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initializes the BCELoss criterion.

        Args:
            reduction: Specifies the reduction to apply: 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes binary cross entropy loss between input and target.

        Args:
            input: Predicted probabilities Tensor.
            target: Ground truth binary target Tensor.

        Returns:
            Output loss Tensor.
        """
        out_data = BCELossOp.compute(input.data, target.data, reduction=self.reduction)
        node = Tensor(data=out_data, op=BCELossOp, inputs=[input, target])
        node.reduction = self.reduction
        return node
