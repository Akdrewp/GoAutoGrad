from GoTorch.backend.native_backend import NDArray as NDArray
from GoTorch.nn.module import Module as Module
from GoTorch.tensor import Tensor as Tensor
from _typeshed import Incomplete

class Linear(Module):
    """Linear Tensor layer with set input and output dimensions.

    Applies an affine linear transformation to the incoming data:
        y = x @ W + b

    Attributes:
        input_dimension: Size of each input feature.
        output_dimension: Size of each output feature.
        linear: Learnable weight Tensor of shape (input_dimension, output_dimension).
        weight: Alias for `self.linear`.
        bias: Learnable bias Tensor of shape (output_dimension,), or None if bias=False.
    """
    input_dimension: Incomplete
    output_dimension: Incomplete
    linear: Incomplete
    weight: Incomplete
    bias: Tensor | None
    def __init__(self, input_dimension: int, output_dimension: int, std: float = 0.1, bias: bool = True) -> None:
        """Initializes the Linear layer with weights and optional bias.

        Args:
            input_dimension: Size of each input feature.
            output_dimension: Size of each output feature.
            std: Standard deviation of normal noise added to initial weights. Defaults to 0.1.
            bias: If True, adds a learnable bias vector. Defaults to True.
        """
    def forward(self, x: Tensor) -> Tensor:
        """Applies the linear transformation to the input Tensor.

        Args:
            x: Input Tensor of shape (..., input_dimension).

        Returns:
            Output Tensor of shape (..., output_dimension).
        """
