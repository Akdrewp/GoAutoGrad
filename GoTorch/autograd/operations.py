from __future__ import annotations

from typing import TYPE_CHECKING

from GoTorch.backend.native_backend import NDArray

if TYPE_CHECKING:
    from GoTorch.tensor import Tensor


def _unbroadcast_gradient(
    grad: "Tensor", target_shape: tuple[int, ...], tensor_cls: type["Tensor"]
) -> "Tensor":
    """Sums gradient across broadcasted dimensions to match target operand shape.

    Args:
        grad: Incoming gradient Tensor.
        target_shape: Shape tuple of the original operand.
        tensor_cls: The Tensor class used to construct the output node.

    Returns:
        Tensor reduced to target_shape via dimensional summation.
    """
    if grad.shape == target_shape:
        return grad

    curr_data: NDArray = grad.data

    # 1. Sum over prepended leading dimensions
    num_leading = len(curr_data.shape) - len(target_shape)
    for _ in range(num_leading):
        curr_data = curr_data.sum(dim=0, keepdim=False)

    # 2. Sum over dimensions that were broadcasted from 1 to D > 1
    for dim_idx, target_dim in enumerate(target_shape):
        if target_dim == 1 and curr_data.shape[dim_idx] > 1:
            curr_data = curr_data.sum(dim=dim_idx, keepdim=True)

    return tensor_cls(curr_data)


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
        tensor_cls = node.__class__
        a, b = node.inputs
        grad_a = _unbroadcast_gradient(out_grad, a.shape, tensor_cls)
        grad_b = _unbroadcast_gradient(out_grad, b.shape, tensor_cls)
        return grad_a, grad_b


class Sub(Operation):
    @classmethod
    def compute(cls, a: NDArray, b: NDArray) -> NDArray:
        return a - b

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        # d(A-B)/dA = 1, d(A-B)/dB = -1
        tensor_cls = node.__class__
        zeros = tensor_cls(out_grad.data.zeros_like())
        neg_grad = zeros - out_grad
        a, b = node.inputs
        grad_a = _unbroadcast_gradient(out_grad, a.shape, tensor_cls)
        grad_b = _unbroadcast_gradient(neg_grad, b.shape, tensor_cls)
        return grad_a, grad_b


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
        return (out_grad.transpose(),)


class ReLU(Operation):
    @classmethod
    def compute(cls, a: NDArray) -> NDArray:
        return a.relu()

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        tensor_cls = node.__class__
        a = node.inputs[0]
        return (tensor_cls(a.data.relu_backward(out_grad.data)),)


class Sigmoid(Operation):
    @classmethod
    def compute(cls, a: NDArray) -> NDArray:
        return a.sigmoid()

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        tensor_cls = node.__class__
        a = node.inputs[0]
        return (tensor_cls(a.data.sigmoid_backward(out_grad.data)),)


class Tanh(Operation):
    @classmethod
    def compute(cls, a: NDArray) -> NDArray:
        return a.tanh()

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        tensor_cls = node.__class__
        a = node.inputs[0]
        return (tensor_cls(a.data.tanh_backward(out_grad.data)),)


class LeakyReLU(Operation):
    @classmethod
    def compute(cls, a: NDArray, alpha: float = 0.01) -> NDArray:
        return a.leaky_relu(alpha=alpha)

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        tensor_cls = node.__class__
        a = node.inputs[0]
        alpha = getattr(node, "alpha", 0.01)
        return (tensor_cls(a.data.leaky_relu_backward(out_grad.data, alpha=alpha)),)


class MSELoss(Operation):
    """Mean squared error loss operation."""

    @classmethod
    def compute(
        cls, prediction: NDArray, target: NDArray, reduction: str = "mean"
    ) -> NDArray:
        """Computes mean squared error loss forward pass on raw NDArrays.

        Args:
            prediction: Predicted values array.
            target: Ground truth target array.
            reduction: Reduction mode ('mean', 'sum', 'none'). Defaults to 'mean'.

        Returns:
            Computed loss NDArray.
        """
        return prediction.mse_loss(target, reduction=reduction)

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        """Returns the gradient Tensor for each input parent.

        Args:
            out_grad: Incoming gradient Tensor from downstream node.
            node: Current loss Tensor node in the computational DAG.

        Returns:
            Tuple of gradient Tensors for parent inputs.
        """
        tensor_cls = node.__class__
        prediction, target = node.inputs
        reduction = getattr(node, "reduction", "mean")
        grad_pred = prediction.data.mse_loss_backward(
            target.data, out_grad.data, reduction=reduction
        )
        grad_target = target.data.mse_loss_backward(
            prediction.data, out_grad.data, reduction=reduction
        )
        return (tensor_cls(grad_pred), tensor_cls(grad_target))


class BCELoss(Operation):
    """Binary cross entropy loss operation."""

    @classmethod
    def compute(
        cls, in_features: NDArray, true_features: NDArray, reduction: str = "mean"
    ) -> NDArray:
        """Computes binary cross entropy loss forward pass on raw NDArrays.

        Args:
            in_features: Predicted probabilities array.
            true_features: Ground truth binary targets array.
            reduction: Reduction mode ('mean', 'sum', 'none'). Defaults to 'mean'.

        Returns:
            Computed loss NDArray.
        """
        return in_features.BCELoss(true_features, reduction=reduction)

    @classmethod
    def gradient(cls, out_grad: "Tensor", node: "Tensor") -> tuple["Tensor", ...]:
        """Returns the gradient Tensor for each input parent.

        Args:
            out_grad: Incoming gradient Tensor from downstream node.
            node: Current loss Tensor node in the computational DAG.

        Returns:
            Tuple of gradient Tensors for parent inputs.
        """
        tensor_cls = node.__class__
        in_features, true_features = node.inputs
        reduction = getattr(node, "reduction", "mean")
        grad_in = in_features.data.bce_loss_backward(
            true_features.data, out_grad.data, reduction=reduction
        )
        zeros = tensor_cls(true_features.data.zeros_like())
        return (tensor_cls(grad_in), zeros)



