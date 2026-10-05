from GoTorch.backend.native_backend import NDArray as NDArray
from GoTorch.tensor import Tensor as Tensor

class Operation:
    @classmethod
    def compute(cls, *inputs: NDArray) -> NDArray:
        """Runs the forward pass on raw NDArrays (calls C/C++ backend)."""
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]:
        """Returns the gradient Tensor for each input parent."""

class Add(Operation):
    @classmethod
    def compute(cls, a: NDArray, b: NDArray) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class Sub(Operation):
    @classmethod
    def compute(cls, a: NDArray, b: NDArray) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class MatMul(Operation):
    @classmethod
    def compute(cls, a: NDArray, b: NDArray) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class Transpose(Operation):
    @classmethod
    def compute(cls, a: NDArray, dim0: int = 0, dim1: int = 1) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class ReLU(Operation):
    @classmethod
    def compute(cls, a: NDArray) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class Sigmoid(Operation):
    @classmethod
    def compute(cls, a: NDArray) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class Tanh(Operation):
    @classmethod
    def compute(cls, a: NDArray) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class LeakyReLU(Operation):
    @classmethod
    def compute(cls, a: NDArray, alpha: float = 0.01) -> NDArray: ...
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]: ...

class MSELoss(Operation):
    """Mean squared error loss operation."""
    @classmethod
    def compute(cls, prediction: NDArray, target: NDArray, reduction: str = 'mean') -> NDArray:
        """Computes mean squared error loss forward pass on raw NDArrays.

        Args:
            prediction: Predicted values array.
            target: Ground truth target array.
            reduction: Reduction mode ('mean', 'sum', 'none'). Defaults to 'mean'.

        Returns:
            Computed loss NDArray.
        """
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]:
        """Returns the gradient Tensor for each input parent.

        Args:
            out_grad: Incoming gradient Tensor from downstream node.
            node: Current loss Tensor node in the computational DAG.

        Returns:
            Tuple of gradient Tensors for parent inputs.
        """

class BCELoss(Operation):
    """Binary cross entropy loss operation."""
    @classmethod
    def compute(cls, in_features: NDArray, true_features: NDArray, reduction: str = 'mean') -> NDArray:
        """Computes binary cross entropy loss forward pass on raw NDArrays.

        Args:
            in_features: Predicted probabilities array.
            true_features: Ground truth binary targets array.
            reduction: Reduction mode ('mean', 'sum', 'none'). Defaults to 'mean'.

        Returns:
            Computed loss NDArray.
        """
    @classmethod
    def gradient(cls, out_grad: Tensor, node: Tensor) -> tuple['Tensor', ...]:
        """Returns the gradient Tensor for each input parent.

        Args:
            out_grad: Incoming gradient Tensor from downstream node.
            node: Current loss Tensor node in the computational DAG.

        Returns:
            Tuple of gradient Tensors for parent inputs.
        """
