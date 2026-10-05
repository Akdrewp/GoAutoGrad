"""Unit tests for GoTorch.nn loss modules (MSELoss, BCELoss)."""

from __future__ import annotations

import math
import pytest

from GoTorch.backend.native_backend import NDArray
from GoTorch.nn.activations import Sigmoid
from GoTorch.nn.layers import Linear
from GoTorch.nn.loss import BCELoss, MSELoss
from GoTorch.nn.optimizer import Adam
from GoTorch.tensor import Tensor


class TestMSELoss:
    """Verifies MSELoss criterion initialization, forward, backward, and optimization."""

    def test_init_default_reduction(self) -> None:
        """Default reduction mode must be 'mean'."""
        loss_fn = MSELoss()
        assert loss_fn.reduction == "mean"

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_init_valid_reductions(self, reduction: str) -> None:
        """Valid reductions ('mean', 'sum', 'none') should be set on the instance."""
        loss_fn = MSELoss(reduction=reduction)
        assert loss_fn.reduction == reduction

    def test_forward_reduction_mean(self) -> None:
        """Computes mean squared error: (1 / N) * sum((input - target)^2)."""
        loss_fn = MSELoss(reduction="mean")
        x = Tensor(NDArray([1.0, 2.0, 3.0], shape=(3,)))
        y = Tensor(NDArray([2.0, 4.0, 1.0], shape=(3,)))

        loss = loss_fn(x, y)
        assert loss.shape == ()
        # Diffs: [-1, -2, 2], squared: [1, 4, 4], sum: 9, mean: 3.0
        assert pytest.approx(loss.data[()], rel=1e-5) == 3.0

    def test_forward_reduction_sum(self) -> None:
        """Computes sum squared error: sum((input - target)^2)."""
        loss_fn = MSELoss(reduction="sum")
        x = Tensor(NDArray([1.0, 2.0, 3.0], shape=(3,)))
        y = Tensor(NDArray([2.0, 4.0, 1.0], shape=(3,)))

        loss = loss_fn(x, y)
        assert loss.shape == ()
        assert pytest.approx(loss.data[()], rel=1e-5) == 9.0

    def test_forward_reduction_none(self) -> None:
        """Computes unreduced squared errors preserving input shape."""
        loss_fn = MSELoss(reduction="none")
        x = Tensor(NDArray([1.0, 2.0, 3.0], shape=(3,)))
        y = Tensor(NDArray([2.0, 4.0, 1.0], shape=(3,)))

        loss = loss_fn(x, y)
        assert loss.shape == (3,)
        assert list(loss.data.data) == pytest.approx([1.0, 4.0, 4.0], rel=1e-5)

    def test_forward_broadcasting(self) -> None:
        """MSELoss broadcasts operands across compatible dimensions: (2, 1) and (1, 3) -> (2, 3)."""
        loss_fn_none = MSELoss(reduction="none")
        loss_fn_mean = MSELoss(reduction="mean")

        x = Tensor(NDArray([1.0, 3.0], shape=(2, 1)))
        y = Tensor(NDArray([0.0, 2.0, 4.0], shape=(1, 3)))

        loss_none = loss_fn_none(x, y)
        assert loss_none.shape == (2, 3)
        expected = [1.0, 1.0, 9.0, 9.0, 1.0, 1.0]
        assert list(loss_none.data.data) == pytest.approx(expected, rel=1e-5)

        loss_mean = loss_fn_mean(x, y)
        assert loss_mean.shape == ()
        assert pytest.approx(loss_mean.data[()], rel=1e-5) == sum(expected) / 6.0

    def test_backward_reduction_mean(self) -> None:
        """Backward pass for reduction='mean' computes dL/dx = (2 / N) * (input - target)."""
        loss_fn = MSELoss(reduction="mean")
        x = Tensor(NDArray([1.0, 2.0, 3.0], shape=(3,)))
        y = Tensor(NDArray([2.0, 4.0, 1.0], shape=(3,)))

        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (3,)
        expected_grad = [(2.0 / 3.0) * -1.0, (2.0 / 3.0) * -2.0, (2.0 / 3.0) * 2.0]
        assert list(x.grad.data) == pytest.approx(expected_grad, rel=1e-5)

    def test_backward_reduction_sum(self) -> None:
        """Backward pass for reduction='sum' computes dL/dx = 2 * (input - target)."""
        loss_fn = MSELoss(reduction="sum")
        x = Tensor(NDArray([1.0, 2.0, 3.0], shape=(3,)))
        y = Tensor(NDArray([2.0, 4.0, 1.0], shape=(3,)))

        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (3,)
        expected_grad = [-2.0, -4.0, 4.0]
        assert list(x.grad.data) == pytest.approx(expected_grad, rel=1e-5)

    def test_backward_reduction_none(self) -> None:
        """Backward pass for reduction='none' with custom upstream gradient."""
        loss_fn = MSELoss(reduction="none")
        x = Tensor(NDArray([1.0, 2.0, 3.0], shape=(3,)))
        y = Tensor(NDArray([2.0, 4.0, 1.0], shape=(3,)))
        out_grad = Tensor(NDArray([1.5, 0.5, 2.0], shape=(3,)))

        loss = loss_fn(x, y)
        loss.backward(out_grad=out_grad)

        assert x.grad is not None
        assert x.grad.shape == (3,)
        expected_grad = [-3.0, -2.0, 8.0]
        assert list(x.grad.data) == pytest.approx(expected_grad, rel=1e-5)

    def test_invalid_reduction_raises_on_forward(self) -> None:
        """Invalid reduction strings raise ValueError on forward execution."""
        loss_fn = MSELoss(reduction="invalid_reduction")
        x = Tensor(NDArray([1.0, 2.0], shape=(2,)))
        y = Tensor(NDArray([2.0, 1.0], shape=(2,)))

        with pytest.raises(ValueError):
            _ = loss_fn(x, y)

    def test_shape_mismatch_raises(self) -> None:
        """Non-broadcastable input and target shapes must raise ValueError."""
        loss_fn = MSELoss()
        x = Tensor(NDArray([1.0, 2.0, 3.0], shape=(3,)))
        y = Tensor(NDArray([1.0, 2.0, 3.0, 4.0], shape=(4,)))

        with pytest.raises(ValueError):
            _ = loss_fn(x, y)

    def test_optimization_loop_integration(self) -> None:
        """Verifies MSELoss integrates in a multi-step Adam optimization loop to reduce loss."""
        linear = Linear(input_dimension=2, output_dimension=1, bias=True)
        optimizer = Adam(linear, lr=0.1, b1=0.9, b2=0.999)
        loss_fn = MSELoss(reduction="mean")

        x = Tensor(NDArray([1.0, 0.5, -0.5, 1.0, 2.0, -1.0], shape=(3, 2)))
        y = Tensor(NDArray([2.0, -1.5, 5.5], shape=(3, 1)))

        initial_loss = loss_fn(linear(x), y).data[()]

        for _ in range(20):
            optimizer.zero_grad()
            pred = linear(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(linear(x), y).data[()]
        assert final_loss < initial_loss
        assert final_loss < 0.1


class TestBCELoss:
    """Verifies BCELoss criterion initialization, forward, backward, and optimization."""

    def test_init_default_reduction(self) -> None:
        """Default reduction mode must be 'mean'."""
        loss_fn = BCELoss()
        assert loss_fn.reduction == "mean"

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_init_valid_reductions(self, reduction: str) -> None:
        """Valid reductions ('mean', 'sum', 'none') should be set on the instance."""
        loss_fn = BCELoss(reduction=reduction)
        assert loss_fn.reduction == reduction

    def test_forward_reduction_mean(self) -> None:
        """Computes mean binary cross entropy: (1 / N) * sum(-(y*log(p) + (1-y)*log(1-p)))."""
        loss_fn = BCELoss(reduction="mean")
        preds = [0.2, 0.8, 0.5]
        targets = [0.0, 1.0, 1.0]
        x = Tensor(NDArray(preds, shape=(3,)))
        y = Tensor(NDArray(targets, shape=(3,)))

        loss = loss_fn(x, y)
        assert loss.shape == ()
        expected = sum(-(t * math.log(p) + (1.0 - t) * math.log(1.0 - p)) for p, t in zip(preds, targets)) / 3.0
        assert pytest.approx(loss.data[()], rel=1e-5) == expected

    def test_forward_reduction_sum(self) -> None:
        """Computes sum binary cross entropy."""
        loss_fn = BCELoss(reduction="sum")
        preds = [0.1, 0.9, 0.4]
        targets = [0.0, 1.0, 0.0]
        x = Tensor(NDArray(preds, shape=(3,)))
        y = Tensor(NDArray(targets, shape=(3,)))

        loss = loss_fn(x, y)
        assert loss.shape == ()
        expected = sum(-(t * math.log(p) + (1.0 - t) * math.log(1.0 - p)) for p, t in zip(preds, targets))
        assert pytest.approx(loss.data[()], rel=1e-5) == expected

    def test_forward_reduction_none(self) -> None:
        """Computes unreduced binary cross entropy preserving input shape."""
        loss_fn = BCELoss(reduction="none")
        preds = [0.25, 0.75]
        targets = [1.0, 0.0]
        x = Tensor(NDArray(preds, shape=(2,)))
        y = Tensor(NDArray(targets, shape=(2,)))

        loss = loss_fn(x, y)
        assert loss.shape == (2,)
        expected = [-(t * math.log(p) + (1.0 - t) * math.log(1.0 - p)) for p, t in zip(preds, targets)]
        assert list(loss.data.data) == pytest.approx(expected, rel=1e-5)

    def test_forward_broadcasting(self) -> None:
        """BCELoss broadcasts operands across compatible dimensions: (2, 1) and (1, 3) -> (2, 3)."""
        loss_fn_none = BCELoss(reduction="none")
        loss_fn_mean = BCELoss(reduction="mean")

        preds = [0.2, 0.8]
        targets = [0.0, 0.5, 1.0]
        x = Tensor(NDArray(preds, shape=(2, 1)))
        y = Tensor(NDArray(targets, shape=(1, 3)))

        loss_none = loss_fn_none(x, y)
        assert loss_none.shape == (2, 3)

        expected = []
        for p in preds:
            for t in targets:
                expected.append(-(t * math.log(p) + (1.0 - t) * math.log(1.0 - p)))

        assert list(loss_none.data.data) == pytest.approx(expected, rel=1e-5)
        loss_mean = loss_fn_mean(x, y)
        assert loss_mean.shape == ()
        assert pytest.approx(loss_mean.data[()], rel=1e-5) == sum(expected) / 6.0

    def test_backward_reduction_mean(self) -> None:
        """Backward pass for reduction='mean' computes dL/dp = (1 / N) * (p - y) / (p * (1 - p))."""
        loss_fn = BCELoss(reduction="mean")
        preds = [0.2, 0.8, 0.5]
        targets = [0.0, 1.0, 1.0]
        x = Tensor(NDArray(preds, shape=(3,)))
        y = Tensor(NDArray(targets, shape=(3,)))

        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (3,)
        expected = [((p - t) / (p * (1.0 - p))) / 3.0 for p, t in zip(preds, targets)]
        assert list(x.grad.data) == pytest.approx(expected, rel=1e-5)

    def test_backward_reduction_sum(self) -> None:
        """Backward pass for reduction='sum' computes dL/dp = (p - y) / (p * (1 - p))."""
        loss_fn = BCELoss(reduction="sum")
        preds = [0.2, 0.8, 0.5]
        targets = [0.0, 1.0, 1.0]
        x = Tensor(NDArray(preds, shape=(3,)))
        y = Tensor(NDArray(targets, shape=(3,)))

        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (3,)
        expected = [(p - t) / (p * (1.0 - p)) for p, t in zip(preds, targets)]
        assert list(x.grad.data) == pytest.approx(expected, rel=1e-5)

    def test_backward_reduction_none(self) -> None:
        """Backward pass for reduction='none' with custom upstream gradient."""
        loss_fn = BCELoss(reduction="none")
        preds = [0.2, 0.8]
        targets = [0.0, 1.0]
        grads = [1.5, 0.5]
        x = Tensor(NDArray(preds, shape=(2,)))
        y = Tensor(NDArray(targets, shape=(2,)))
        out_grad = Tensor(NDArray(grads, shape=(2,)))

        loss = loss_fn(x, y)
        loss.backward(out_grad=out_grad)

        assert x.grad is not None
        assert x.grad.shape == (2,)
        expected = [g * (p - t) / (p * (1.0 - p)) for p, t, g in zip(preds, targets, grads)]
        assert list(x.grad.data) == pytest.approx(expected, rel=1e-5)

    def test_invalid_reduction_raises_on_forward(self) -> None:
        """Invalid reduction strings raise ValueError on forward execution."""
        loss_fn = BCELoss(reduction="unknown_reduction")
        x = Tensor(NDArray([0.5, 0.5], shape=(2,)))
        y = Tensor(NDArray([1.0, 0.0], shape=(2,)))

        with pytest.raises(ValueError):
            _ = loss_fn(x, y)

    def test_shape_mismatch_raises(self) -> None:
        """Non-broadcastable input and target shapes must raise ValueError."""
        loss_fn = BCELoss()
        x = Tensor(NDArray([0.5] * 3, shape=(3,)))
        y = Tensor(NDArray([1.0] * 5, shape=(5,)))

        with pytest.raises(ValueError):
            _ = loss_fn(x, y)

    def test_optimization_loop_integration(self) -> None:
        """Verifies BCELoss integrates in an Adam optimization loop with Sigmoid activation."""
        linear = Linear(input_dimension=2, output_dimension=1, bias=True)
        sigmoid = Sigmoid()
        optimizer = Adam(linear, lr=0.1, b1=0.9, b2=0.999)
        loss_fn = BCELoss(reduction="mean")

        x = Tensor(NDArray([1.0, 1.0, -1.0, -1.0, 1.5, 0.5, -1.5, -0.5], shape=(4, 2)))
        y = Tensor(NDArray([1.0, 0.0, 1.0, 0.0], shape=(4, 1)))

        initial_loss = loss_fn(sigmoid(linear(x)), y).data[()]

        for _ in range(20):
            optimizer.zero_grad()
            pred = sigmoid(linear(x))
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(sigmoid(linear(x)), y).data[()]
        assert final_loss < initial_loss
        assert final_loss < 0.2
