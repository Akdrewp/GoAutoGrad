from __future__ import annotations

from typing import TYPE_CHECKING

from GoTorch.backend.native_backend import NDArray

if TYPE_CHECKING:
    from GoTorch.tensor import Tensor


# Specified operations that take in NDArrays and return results
class Operation:
    @classmethod
    def compute(cls, *inputs: NDArray) -> NDArray:
        """Runs the forward pass on raw NDArrays (calls C/C++ backend)."""
        raise NotImplementedError

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        """Returns the gradient Tensor for each input parent."""
        raise NotImplementedError


class Add(Operation):
    @classmethod
    def compute(cls, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        # d(A+B)/dA = 1, d(A+B)/dB = 1
        return out_grad, out_grad


class Sub(Operation):
    @classmethod
    def compute(cls, a: NDArray, b: NDArray) -> NDArray:
        return a - b

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        raise NotImplementedError


class MatMul(Operation):
    @classmethod
    def compute(cls, a: NDArray, b: NDArray) -> NDArray:
        return a.matmul(b)

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        # out = A @ B -> grad_A = out_grad @ B.T, grad_B = A.T @ out_grad
        a, b = node.inputs
        return out_grad @ b.transpose(), a.transpose() @ out_grad


class Transpose(Operation):
    @classmethod
    def compute(cls, a: NDArray, dim0: int = 0, dim1: int = 1) -> NDArray:
        return a.transpose(dim0, dim1)

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        raise NotImplementedError
