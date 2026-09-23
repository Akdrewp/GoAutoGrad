"""Common layers to be used within a module."""

from __future__ import annotations

import random

from GoTorch.backend.native_backend import NDArray
from GoTorch.nn.module import Module
from GoTorch.tensor import Tensor


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

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        std: float = 0.1,
        bias: bool = True,
    ) -> None:
        """Initializes the Linear layer with weights and optional bias.

        Args:
            input_dimension: Size of each input feature.
            output_dimension: Size of each output feature.
            std: Standard deviation of normal noise added to initial weights. Defaults to 0.1.
            bias: If True, adds a learnable bias vector. Defaults to True.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # Create initial weight with shape (input, output)
        # Add 0s matrix that has std = std to initial matrix
        # Now we have a matrix full of 1s with std = 0.1
        total_weights = input_dimension * output_dimension
        weight_values = [random.gauss(1.0, std) for _ in range(total_weights)]
        self.linear = Tensor(
            NDArray(weight_values, shape=(input_dimension, output_dimension))
        )
        self.weight = self.linear

        # Create bias matrix of 0s
        if bias:
            self.bias: Tensor | None = Tensor(
                NDArray([0.0] * output_dimension, shape=(output_dimension,))
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Applies the linear transformation to the input Tensor.

        Args:
            x: Input Tensor of shape (..., input_dimension).

        Returns:
            Output Tensor of shape (..., output_dimension).
        """
        result = x @ self.linear
        if self.bias is not None:
            result = result + self.bias

        return result
