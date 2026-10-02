"""Activation layer wrappers."""

from __future__ import annotations

from GoTorch.nn.module import Module
from GoTorch.tensor import Tensor


class ReLU(Module):
    """Applies the Rectified Linear Unit (ReLU) function element-wise.

    ReLU(x) = max(0, x)
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies ReLU activation to the input Tensor.

        Args:
            x: Input Tensor.

        Returns:
            Output Tensor with ReLU applied.
        """
        return x.relu()


class Sigmoid(Module):
    """Applies the Sigmoid function element-wise.

    Sigmoid(x) = 1 / (1 + exp(-x))
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies Sigmoid activation to the input Tensor.

        Args:
            x: Input Tensor.

        Returns:
            Output Tensor with Sigmoid applied.
        """
        return x.sigmoid()


class Tanh(Module):
    """Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies Tanh activation to the input Tensor.

        Args:
            x: Input Tensor.

        Returns:
            Output Tensor with Tanh applied.
        """
        return x.tanh()


class LeakyReLU(Module):
    """Applies the Leaky Rectified Linear Unit function element-wise.

    LeakyReLU(x) = max(0, x) + alpha * min(0, x)

    Attributes:
        alpha: Slope of the activation for x < 0. Defaults to 0.01.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Initializes the LeakyReLU layer.

        Args:
            alpha: Controls the angle of the negative slope. Defaults to 0.01.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """Applies LeakyReLU activation to the input Tensor.

        Args:
            x: Input Tensor.

        Returns:
            Output Tensor with LeakyReLU applied.
        """
        return x.leaky_relu(alpha=self.alpha)
