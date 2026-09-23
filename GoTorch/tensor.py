from __future__ import annotations

from GoTorch.autograd.autograd_engine import AutogradEngine
from GoTorch.autograd.operations import Add, MatMul, Sub, Transpose
from GoTorch.backend.native_backend import NDArray


# The Autograd Graph Node (Frontend / Python User API)
class Tensor:
    """The autograd graph node representing a multidimensional array.

    Attributes:
        data: The underlying NDArray storing the tensor elements.
        grad: The accumulated gradient (NDArray) computed during the backward
            pass, or None if backward has not yet been executed.
        op: The Operation class that produced this tensor, or None for leaf
            nodes.
        inputs: List of parent Tensor nodes passed to the op.
    """

    def __init__(
        self,
        data: NDArray,
        op: type | None = None,
        inputs: list[Tensor] | None = None,
    ) -> None:
        """Initializes a Tensor node.

        Args:
            data: The computed NDArray.
            op: The operation that created this Tensor.
            inputs: List of parent Tensor nodes passed to the op.
        """
        self.data = data
        self.grad = None
        self.op = op
        self.inputs = inputs or []

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape dimensions of the underlying NDArray."""
        return self.data.shape

    def backward(self, out_grad: Tensor | None = None) -> None:
        """Performs backward pass on tensor via post-order traversal.

        Args:
            out_grad: The activation gradient input to this tensor. Defaults
                to array of 1s with shape=self.data.shape.
        """
        AutogradEngine.backward(self, out_grad)

    def __add__(self, other: Tensor) -> Tensor:
        """Constructs an Add node in the computational DAG.

        Args:
            other: Tensor to add to this tensor.

        Returns:
            A Tensor result of the addition added to the graph.

        Raises:
            TypeError: If other is not a Tensor.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for +: {type(other)}")

        out_data = Add.compute(self.data, other.data)
        return Tensor(data=out_data, op=Add, inputs=[self, other])

    def __radd__(self, other: Tensor) -> Tensor:
        """Constructs an Add node in the computational DAG for right-hand addition.

        Args:
            other: Tensor to add to this tensor.

        Returns:
            A Tensor result of the addition added to the graph.

        Raises:
            TypeError: If other is not a Tensor.
        """
        return self.__add__(other)

    def __sub__(self, other: Tensor) -> Tensor:
        """Constructs a Sub node in the computational DAG.

        Args:
            other: Tensor to subtract from this tensor.

        Returns:
            A Tensor result of the subtraction added to the graph.

        Raises:
            TypeError: If other is not a Tensor.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for -: {type(other)}")

        out_data = Sub.compute(self.data, other.data)
        return Tensor(data=out_data, op=Sub, inputs=[self, other])

    def __rsub__(self, other: Tensor) -> Tensor:
        """Constructs a Sub node in the computational DAG for right-hand subtraction.

        Args:
            other: Tensor to subtract this tensor from.

        Returns:
            A Tensor result of the subtraction added to the graph.

        Raises:
            TypeError: If other is not a Tensor.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for -: {type(other)}")

        out_data = Sub.compute(other.data, self.data)
        return Tensor(data=out_data, op=Sub, inputs=[other, self])

    def __matmul__(self, other: Tensor) -> Tensor:
        """Constructs a MatMul node in the computational DAG.

        Args:
            other: Tensor to multiply with.

        Returns:
            A Tensor result of the multiplication added to the graph.

        Raises:
            TypeError: If other is not a Tensor.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for @: {type(other)}")

        out_data = MatMul.compute(self.data, other.data)
        return Tensor(data=out_data, op=MatMul, inputs=[self, other])

    def transpose(self, dim0: int = 0, dim1: int = 1) -> Tensor:
        """Returns a transposed Tensor sharing the underlying storage view.

        Args:
            dim0: First dimension to swap. Defaults to 0.
            dim1: Second dimension to swap. Defaults to 1.

        Returns:
            A transposed Tensor viewing the underlying storage.
        """
        out_data = Transpose.compute(self.data, dim0, dim1)
        return Tensor(data=out_data, op=Transpose, inputs=[self])

    def __repr__(self) -> str:
        """Returns string representation of the Tensor."""
        return f"Tensor({self.data})"
