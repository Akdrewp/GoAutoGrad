from GoTorch.nn.module import Module as Module
from GoTorch.tensor import Tensor as Tensor
from _typeshed import Incomplete

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

class LeakyReLU(Module):
    """Applies the Leaky Rectified Linear Unit function element-wise.

    LeakyReLU(x) = max(0, x) + alpha * min(0, x)

    Attributes:
        alpha: Slope of the activation for x < 0. Defaults to 0.01.
    """
    alpha: Incomplete
    def __init__(self, alpha: float = 0.01) -> None:
        """Initializes the LeakyReLU layer.

        Args:
            alpha: Controls the angle of the negative slope. Defaults to 0.01.
        """
    def forward(self, x: Tensor) -> Tensor:
        """Applies LeakyReLU activation to the input Tensor.

        Args:
            x: Input Tensor.

        Returns:
            Output Tensor with LeakyReLU applied.
        """
