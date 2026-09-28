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
