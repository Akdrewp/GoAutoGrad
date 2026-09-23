from __future__ import annotations

import pytest
import torch

from GoTorch.backend.native_backend import NDArray
from GoTorch.nn.layers import Linear
from GoTorch.nn.module import Module
from GoTorch.nn.optimizer import Adam, Optimizer
from GoTorch.tensor import Tensor


class TestOptimizerBase:
    """Verifies basic Optimizer functionality."""

    def test_optimizer_init_module(self) -> None:
        """Verifies optimizer extracts parameters from a Module."""
        layer = Linear(2, 3, bias=True)
        opt = Optimizer(layer, lr=0.01, b1=0.9, b2=0.999)

        assert len(opt.parameters) == 2
        assert layer.linear in opt.parameters
        assert layer.bias in opt.parameters
        assert opt.lr == 0.01
        assert opt.b1 == 0.9
        assert opt.b2 == 0.999

    def test_optimizer_init_list(self) -> None:
        """Verifies optimizer accepts a list of Tensors."""
        t1 = Tensor(NDArray([1.0], shape=(1,)))
        t2 = Tensor(NDArray([2.0], shape=(1,)))
        opt = Optimizer([t1, t2], lr=0.05, b1=0.9, b2=0.999)

        assert opt.parameters == [t1, t2]
        assert opt.lr == 0.05
        assert opt.b1 == 0.9
        assert opt.b2 == 0.999

    def test_optimizer_zero_grad(self) -> None:
        """Verifies zero_grad clears gradients on managed parameters."""
        t = Tensor(NDArray([1.0, 2.0], shape=(2,)))
        t.grad = NDArray([0.5, 0.5], shape=(2,))
        opt = Optimizer([t], lr=0.01, b1=0.9, b2=0.999)

        assert t.grad is not None
        opt.zero_grad()
        assert t.grad is None

    def test_optimizer_step_not_implemented(self) -> None:
        """Verifies base class step() raises NotImplementedError."""
        t = Tensor(NDArray([1.0], shape=(1,)))
        opt = Optimizer([t], lr=0.01, b1=0.9, b2=0.999)
        with pytest.raises(NotImplementedError):
            opt.step()


class TestAdamOptimizer:
    """Verifies Adam optimizer against mathematical reference and PyTorch."""

    def test_adam_step_matches_pytorch(self) -> None:
        """Verifies Adam parameter updates match PyTorch torch.optim.Adam."""
        init_weights = [1.0, 2.0, 3.0, 4.0]
        init_grads = [0.1, -0.2, 0.5, -0.05]

        # GoTorch Adam
        p_gt = Tensor(NDArray(init_weights, shape=(2, 2)))
        p_gt.grad = NDArray(init_grads, shape=(2, 2))
        opt_gt = Adam([p_gt], lr=0.01, b1=0.9, b2=0.999, eps=1e-8)
        opt_gt.step()

        # PyTorch Adam
        p_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        opt_pt = torch.optim.Adam([p_pt], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        p_pt.grad = torch.tensor([[0.1, -0.2], [0.5, -0.05]], dtype=torch.float32)
        opt_pt.step()

        # Assert parameter data matches PyTorch after 1 step
        assert list(p_gt.data.data) == pytest.approx(p_pt.detach().flatten().tolist(), rel=1e-5)

    def test_adam_multiple_steps_matches_pytorch(self) -> None:
        """Verifies multi-step momentum and second moment tracking match PyTorch."""
        init_weights = [0.5, -1.0]

        p_gt = Tensor(NDArray(init_weights, shape=(2,)))
        opt_gt = Adam([p_gt], lr=0.05, b1=0.85, b2=0.95, eps=1e-8)

        p_pt = torch.tensor(init_weights, dtype=torch.float32, requires_grad=True)
        opt_pt = torch.optim.Adam([p_pt], lr=0.05, betas=(0.85, 0.95), eps=1e-8)

        # Simulate 3 steps with varying gradients
        grads = [
            [0.2, -0.3],
            [0.15, -0.25],
            [0.1, -0.1],
        ]

        for g in grads:
            p_gt.grad = NDArray(g, shape=(2,))
            opt_gt.step()

            p_pt.grad = torch.tensor(g, dtype=torch.float32)
            opt_pt.step()

            # Assert parameter data matches PyTorch at every step
            assert list(p_gt.data.data) == pytest.approx(p_pt.detach().tolist(), rel=1e-5)

    def test_adam_training_loop_with_linear_layer(self) -> None:
        """Verifies Adam updates weights of a Linear module in an optimization loop against PyTorch."""
        layer = Linear(2, 1, bias=False)
        layer.linear = Tensor(NDArray([2.0, 3.0], shape=(2, 1)))

        x = Tensor(NDArray([1.0, 1.0], shape=(1, 2)))
        opt = Adam(layer, lr=0.1, b1=0.9, b2=0.999, eps=1e-8)

        # PyTorch reference model with identical initial weights and input
        layer_pt = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            layer_pt.weight.copy_(torch.tensor([[2.0, 3.0]]))
        x_pt = torch.tensor([[1.0, 1.0]])
        opt_pt = torch.optim.Adam(layer_pt.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)

        # Initial prediction = [2.0*1.0 + 3.0*1.0] = [5.0]
        initial_val = (layer(x)).data[0, 0]

        for _ in range(5):
            opt.zero_grad()
            out = layer(x)
            out.backward()
            opt.step()

            opt_pt.zero_grad()
            out_pt = layer_pt(x_pt)
            out_pt.backward(torch.ones_like(out_pt))
            opt_pt.step()

        # Assert final updated weights match PyTorch
        expected_weights = layer_pt.weight.detach().flatten().tolist()
        assert list(layer.linear.data.data) == pytest.approx(expected_weights, rel=1e-5)

        # Assert final forward prediction matches PyTorch
        expected_output = layer_pt(x_pt).item()
        assert (layer(x)).data[0, 0] == pytest.approx(expected_output, rel=1e-5)

        # Assert output decreased monotonically from initial value
        assert (layer(x)).data[0, 0] < initial_val
