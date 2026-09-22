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

    def __matmul__(self, other: Tensor) -> Tensor:
        """Constructs a MatMul node in the computational DAG

        Args:
            other: Tensor to multiple with

        Returns: 
            A Tensor result of the multiplication added to the graph
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for @: {type(other)}")

        out_data = MatMul.compute(self.data, other.data)
        return Tensor(data=out_data, op=MatMul, inputs=[self, other])

    def transpose(self, dim0: int = 0, dim1: int = 1) -> Tensor:
        """Returns a transposed Tensor sharing the underlying storage view
        
        

        """
        # For leaf/intermediate nodes in gradients, transpose delegates to backend
        return Tensor(data=self.data.transpose(dim0, dim1))
