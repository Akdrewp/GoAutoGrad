from __future__ import annotations

import pytest

from GoTorch.backend.native_backend import NDArray
from GoTorch.nn.layers import Linear
from GoTorch.nn.module import Module
from GoTorch.nn.optimizer import Adam
from GoTorch.tensor import Tensor


class TestLinearLayer:
    """Verifies Linear layer initialization, forward transformation, autograd, and fitting."""

    def test_linear_init(self) -> None:
        """Verifies weight and bias shapes and layer attributes."""
        layer = Linear(input_dimension=3, output_dimension=2, std=0.05, bias=True)

        assert isinstance(layer, Module)
        assert layer.input_dimension == 3
        assert layer.output_dimension == 2
        assert layer.linear.shape == (3, 2)
        assert layer.weight is layer.linear
        assert layer.bias is not None
        assert layer.bias.shape == (2,)

    def test_linear_init_no_bias(self) -> None:
        """Verifies layer initialization when bias=False."""
        layer = Linear(input_dimension=4, output_dimension=3, bias=False)

        assert layer.bias is None
        assert layer.linear.shape == (4, 3)

    def test_linear_forward_with_bias(self) -> None:
        """Verifies forward computation y = x @ W + b."""
        layer = Linear(input_dimension=2, output_dimension=2, bias=True)
        # Override weights and bias with known values for deterministic assertion
        layer.linear = Tensor(NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        layer.bias = Tensor(NDArray([0.5, -0.5], shape=(2,)))

        x = Tensor(NDArray([1.0, 1.0, 2.0, 0.0], shape=(2, 2)))
        out = layer(x)

        # x @ W = [[1, 1], [2, 0]] @ [[1, 2], [3, 4]]
        # Row 0: [1*1 + 1*3, 1*2 + 1*4] = [4.0, 6.0]
        # Row 1: [2*1 + 0*3, 2*2 + 0*4] = [2.0, 4.0]
        # + b:
        # Row 0: [4.5, 5.5]
        # Row 1: [2.5, 3.5]
        assert out.shape == (2, 2)
        assert list(out.data.data) == pytest.approx([4.5, 5.5, 2.5, 3.5])

    def test_linear_forward_no_bias(self) -> None:
        """Verifies forward computation without bias: y = x @ W."""
        layer = Linear(input_dimension=2, output_dimension=1, bias=False)
        layer.linear = Tensor(NDArray([2.0, 3.0], shape=(2, 1)))

        x = Tensor(NDArray([1.0, 2.0], shape=(1, 2)))
        out = layer(x)

        assert out.shape == (1, 1)
        assert list(out.data.data) == pytest.approx([8.0])

    def test_linear_backward(self) -> None:
        """Verifies gradient propagation through Linear layer."""
        layer = Linear(input_dimension=2, output_dimension=2, bias=True)
        layer.linear = Tensor(NDArray([1.0, 0.0, 0.0, 1.0], shape=(2, 2)))
        layer.bias = Tensor(NDArray([0.0, 0.0], shape=(2,)))

        x = Tensor(NDArray([2.0, 3.0], shape=(1, 2)))
        out = layer(x)
        out.backward()

        # out_grad = [[1.0, 1.0]]
        # layer.bias.grad = [1.0, 1.0]
        # layer.linear.grad = x.T @ out_grad = [[2], [3]] @ [[1, 1]] = [[2, 2], [3, 3]]
        # x.grad = out_grad @ W.T = [[1, 1]] @ [[1, 0], [0, 1]] = [[1, 1]]
        assert layer.bias.grad is not None
        assert list(layer.bias.grad.data) == pytest.approx([1.0, 1.0])
        assert list(layer.linear.grad.data) == pytest.approx([2.0, 2.0, 3.0, 3.0])
        assert list(x.grad.data) == pytest.approx([1.0, 1.0])

    def test_two_layer_overfit_to_square(self) -> None:
        """Verifies a two-layer Linear network fits the square of input tensors."""
        class TwoLayerNet(Module):
            def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
                super().__init__()
                self.fc1 = Linear(in_features, hidden_features, bias=True)
                self.fc2 = Linear(hidden_features, out_features, bias=True)

            def forward(self, x: Tensor) -> Tensor:
                return self.fc2(self.fc1(x))

        # Input tensor
        x = Tensor(NDArray([1.0, 2.0], shape=(1, 2)))
        # Target is the square of the inputs: [1.0^2, 2.0^2] = [1.0, 4.0]
        target = Tensor(NDArray([1.0, 4.0], shape=(1, 2)))

        net = TwoLayerNet(in_features=2, hidden_features=4, out_features=2)
        optimizer = Adam(net, lr=0.1, b1=0.9, b2=0.999)

        initial_pred = net(x)
        initial_diff = initial_pred - target
        initial_loss = (initial_diff @ initial_diff.transpose(0, 1)).data[0, 0]

        # Training loop optimizing the difference between predictions and squared inputs
        for _ in range(150):
            optimizer.zero_grad()
            pred = net(x)
            diff = pred - target
            loss = diff @ diff.transpose(0, 1)
            loss.backward()
            optimizer.step()

        final_pred = net(x)
        final_diff = final_pred - target
        final_loss = (final_diff @ final_diff.transpose(0, 1)).data[0, 0]

        # Assert loss converged near 0 and predictions accurately match input squares
        assert final_loss < 1e-4
        assert final_loss < initial_loss * 1e-4
        assert list(final_pred.data.data) == pytest.approx([1.0, 4.0], abs=1e-2)
