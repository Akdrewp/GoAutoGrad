from __future__ import annotations

from GoTorch.autograd.autograd_engine import AutogradEngine
from GoTorch.autograd.operations import Add, MatMul
from GoTorch.backend.ndarray import NDArray


# The Autograd Graph Node (Frontend / Python User API)
class Tensor:
    def __init__(self, data: NDArray, op=None, inputs=None):
        self.data = data  # The computed NDArray
        self.grad = None  # Accumulated gradient (another NDArray)
        self.op = op  # The operation that created this Tensor
        self.inputs = inputs or []  # List of parent Tensor nodes passed to the op

    def backward(self, out_grad: "Tensor"):
        """Performs backward pass on tensor via post-order traversal.

        Args:
            out_grad: The activation gradient input to this tensor.
            Defaults to array of 1s with shape=self.data.shape
        """
        AutogradEngine.backward(self, out_grad)
