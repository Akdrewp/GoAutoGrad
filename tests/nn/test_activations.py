from __future__ import annotations

import math
import pytest

from GoTorch.autograd.operations import LeakyReLU as LeakyReLUOp
from GoTorch.autograd.operations import ReLU as ReLUOp
from GoTorch.autograd.operations import Sigmoid as SigmoidOp
from GoTorch.autograd.operations import Tanh as TanhOp
from GoTorch.nn.activations import LeakyReLU, ReLU, Sigmoid, Tanh
from GoTorch.nn.layers import Linear
from GoTorch.nn.module import Module
from GoTorch.tensor import Tensor


class TestActivationForward:
    """Verifies forward pass transformation for activation layers."""

    def test_relu_forward(self, backend) -> None:
        """Verifies ReLU rectification: max(0, x)."""
        layer = ReLU()
        raw = [-2.0, -0.5, 0.0, 1.5, 3.0]
        x = Tensor(backend.NDArray(raw, shape=(5,)))
        out = layer(x)

        assert isinstance(out, Tensor)
        assert out.shape == (5,)
        assert list(out.data.data) == [0.0, 0.0, 0.0, 1.5, 3.0]

    def test_sigmoid_forward(self, backend) -> None:
        """Verifies Sigmoid function: 1 / (1 + exp(-x))."""
        layer = Sigmoid()
        raw = [-2.0, 0.0, 2.0]
        x = Tensor(backend.NDArray(raw, shape=(3,)))
        out = layer(x)

        assert isinstance(out, Tensor)
        assert out.shape == (3,)
        expected = [1.0 / (1.0 + math.exp(-v)) for v in raw]
        for act, exp in zip(out.data.data, expected):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_tanh_forward(self, backend) -> None:
        """Verifies Tanh function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))."""
        layer = Tanh()
        raw = [-1.5, 0.0, 1.5]
        x = Tensor(backend.NDArray(raw, shape=(3,)))
        out = layer(x)

        assert isinstance(out, Tensor)
        assert out.shape == (3,)
        expected = [math.tanh(v) for v in raw]
        for act, exp in zip(out.data.data, expected):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_leaky_relu_default_alpha_forward(self, backend) -> None:
        """Verifies LeakyReLU with default alpha=0.01."""
        layer = LeakyReLU()
        raw = [-20.0, -10.0, 0.0, 5.0]
        x = Tensor(backend.NDArray(raw, shape=(4,)))
        out = layer(x)

        assert isinstance(out, Tensor)
        assert out.shape == (4,)
        assert list(out.data.data) == pytest.approx([-0.2, -0.1, 0.0, 5.0], rel=1e-5)

    def test_leaky_relu_custom_alpha_forward(self, backend) -> None:
        """Verifies LeakyReLU with custom alpha."""
        layer = LeakyReLU(alpha=0.1)
        assert layer.alpha == 0.1

        raw = [-10.0, 0.0, 2.0]
        x = Tensor(backend.NDArray(raw, shape=(3,)))
        out = layer(x)

        assert list(out.data.data) == pytest.approx([-1.0, 0.0, 2.0], rel=1e-5)


class TestActivationParameters:
    """Verifies that activation layers contain no learnable parameters."""

    @pytest.mark.parametrize("act_cls", [ReLU, Sigmoid, Tanh, LeakyReLU])
    def test_activation_parameters_empty(self, act_cls) -> None:
        """Activation layers have no weights or biases; parameters() must return empty list."""
        layer = act_cls()
        assert isinstance(layer, Module)
        assert layer.parameters() == []


class TestActivationAutograd:
    """Verifies gradient propagation through activation layers and Tensor methods."""

    def test_relu_backward(self, backend) -> None:
        """Verifies ReLU backward pass propagates gradients where x > 0."""
        layer = ReLU()
        raw_x = [-2.0, -0.5, 0.0, 1.0, 3.0]
        x = Tensor(backend.NDArray(raw_x, shape=(5,)))
        out = layer(x)

        grad_out = Tensor(backend.NDArray([10.0, 20.0, 30.0, 40.0, 50.0], shape=(5,)))
        out.backward(grad_out)

        assert x.grad is not None
        assert list(x.grad.data) == [0.0, 0.0, 0.0, 40.0, 50.0]

    def test_sigmoid_backward(self, backend) -> None:
        """Verifies Sigmoid backward pass propagates dy * s * (1 - s)."""
        layer = Sigmoid()
        raw_x = [-2.0, 0.0, 2.0]
        x = Tensor(backend.NDArray(raw_x, shape=(3,)))
        out = layer(x)

        grad_out = Tensor(backend.NDArray([1.0, 2.0, 1.0], shape=(3,)))
        out.backward(grad_out)

        assert x.grad is not None
        # At x=0, s=0.5 -> dy * 0.25 -> 2.0 * 0.25 = 0.5
        expected = [
            1.0 * (1.0 / (1.0 + math.exp(2.0))) * (1.0 - 1.0 / (1.0 + math.exp(2.0))),
            0.5,
            1.0 * (1.0 / (1.0 + math.exp(-2.0))) * (1.0 - 1.0 / (1.0 + math.exp(-2.0))),
        ]
        for act, exp in zip(x.grad.data, expected):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_tanh_backward(self, backend) -> None:
        """Verifies Tanh backward pass propagates dy * (1 - tanh(x)^2)."""
        layer = Tanh()
        raw_x = [-1.0, 0.0, 1.0]
        x = Tensor(backend.NDArray(raw_x, shape=(3,)))
        out = layer(x)

        grad_out = Tensor(backend.NDArray([2.0, 3.0, 4.0], shape=(3,)))
        out.backward(grad_out)

        assert x.grad is not None
        # At x=0, tanh(0)=0 -> dy * 1.0 = 3.0
        expected = [
            2.0 * (1.0 - math.tanh(-1.0) ** 2),
            3.0,
            4.0 * (1.0 - math.tanh(1.0) ** 2),
        ]
        for act, exp in zip(x.grad.data, expected):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_leaky_relu_backward_custom_alpha(self, backend) -> None:
        """Verifies LeakyReLU backward pass with custom alpha."""
        layer = LeakyReLU(alpha=0.2)
        raw_x = [-4.0, 0.0, 5.0]
        x = Tensor(backend.NDArray(raw_x, shape=(3,)))
        out = layer(x)

        grad_out = Tensor(backend.NDArray([10.0, 10.0, 10.0], shape=(3,)))
        out.backward(grad_out)

        assert x.grad is not None
        assert list(x.grad.data) == pytest.approx([2.0, 2.0, 10.0], rel=1e-5)

    def test_tensor_activation_methods_dag_nodes(self, backend) -> None:
        """Verifies direct Tensor activation methods create appropriate DAG Operation nodes."""
        x = Tensor(backend.NDArray([1.0, -1.0], shape=(2,)))

        r = x.relu()
        assert r.op is ReLUOp
        assert r.inputs == [x]

        s = x.sigmoid()
        assert s.op is SigmoidOp
        assert s.inputs == [x]

        t = x.tanh()
        assert t.op is TanhOp
        assert t.inputs == [x]

        lr = x.leaky_relu(alpha=0.05)
        assert lr.op is LeakyReLUOp
        assert lr.inputs == [x]


class TestActivationPipelineIntegration:
    """Verifies integration of activation layers inside composite multi-layer Modules."""

    def test_linear_relu_linear_pipeline_forward(self, backend) -> None:
        """Verifies end-to-end forward propagation through Linear -> ReLU -> Linear."""
        class MLP(Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = Linear(2, 3, bias=True)
                self.act = ReLU()
                self.fc2 = Linear(3, 1, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.fc2(self.act(self.fc1(x)))

        net = MLP()
        # Set deterministic weights for exact validation
        net.fc1.linear = Tensor(backend.NDArray([1.0, -1.0, 2.0, 0.5, -0.5, 1.0], shape=(2, 3)))
        net.fc1.bias = Tensor(backend.NDArray([0.0, 0.0, 0.0], shape=(3,)))
        net.fc2.linear = Tensor(backend.NDArray([1.0, 2.0, 1.0], shape=(3, 1)))

        x = Tensor(backend.NDArray([1.0, 2.0], shape=(1, 2)))
        out = net(x)

        # fc1: [1.0, 2.0] @ [[1.0, -1.0, 2.0], [0.5, -0.5, 1.0]] = [2.0, -2.0, 4.0]
        # relu: [2.0, 0.0, 4.0]
        # fc2: [2.0, 0.0, 4.0] @ [[1.0], [2.0], [1.0]] = 2*1 + 0*2 + 4*1 = 6.0
        assert out.shape == (1, 1)
        assert list(out.data.data) == pytest.approx([6.0])

    def test_linear_relu_linear_pipeline_backward(self, backend) -> None:
        """Verifies gradient flows backward through ReLU to upstream Linear parameters and input."""
        class MLP(Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = Linear(2, 2, bias=True)
                self.act = ReLU()
                self.fc2 = Linear(2, 1, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.fc2(self.act(self.fc1(x)))

        net = MLP()
        net.fc1.linear = Tensor(backend.NDArray([1.0, -1.0, 1.0, 1.0], shape=(2, 2)))
        net.fc1.bias = Tensor(backend.NDArray([0.0, 0.0], shape=(2,)))
        net.fc2.linear = Tensor(backend.NDArray([2.0, 3.0], shape=(2, 1)))

        x = Tensor(backend.NDArray([1.0, 1.0], shape=(1, 2)))
        out = net(x)

        # Backward from out with dy = 1.0
        out_grad = Tensor(backend.NDArray([1.0], shape=(1, 1)))
        out.backward(out_grad)

        # Verify all parent nodes and learnable weights have accumulated gradients
        assert net.fc2.linear.grad is not None
        assert net.fc1.linear.grad is not None
        assert net.fc1.bias.grad is not None
        assert x.grad is not None

    def test_pipeline_parameter_collection(self) -> None:
        """Verifies net.parameters() excludes activation layers and only returns learnable weights."""
        class ComplexMLP(Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = Linear(4, 8, bias=True)
                self.act1 = ReLU()
                self.fc2 = Linear(8, 4, bias=False)
                self.act2 = Tanh()
                self.fc3 = Linear(4, 1, bias=True)
                self.act3 = Sigmoid()

            def forward(self, x: Tensor) -> Tensor:
                return self.act3(self.fc3(self.act2(self.fc2(self.act1(self.fc1(x))))))

        net = ComplexMLP()
        params = net.parameters()

        # fc1 has weight + bias (2)
        # fc2 has weight only (1)
        # fc3 has weight + bias (2)
        # Activations have 0
        # Total = 5 parameters
        assert len(params) == 5
        assert net.fc1.linear in params
        assert net.fc1.bias in params
        assert net.fc2.linear in params
        assert net.fc3.linear in params
        assert net.fc3.bias in params
