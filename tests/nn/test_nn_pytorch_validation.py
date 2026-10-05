from __future__ import annotations

import random
from typing import Callable
import pytest
import torch

from GoTorch.backend.native_backend import NDArray
from GoTorch.nn.activations import LeakyReLU, ReLU, Sigmoid, Tanh
from GoTorch.nn.layers import Linear
from GoTorch.nn.loss import BCELoss, MSELoss
from GoTorch.nn.module import Module, Sequential
from GoTorch.nn.optimizer import Adam, Optimizer
from GoTorch.tensor import Tensor


def sync_linear_weights(gt_linear: Linear, pt_linear: torch.nn.Linear) -> None:
    """Synchronizes weights and biases from GoTorch Linear to PyTorch nn.Linear.

    GoTorch Linear weight shape: (in_features, out_features), computing: y = x @ W + b
    PyTorch nn.Linear weight shape: (out_features, in_features), computing: y = x @ W.T + b
    Therefore: pt_linear.weight = gt_linear.linear.T
    """
    in_dim = gt_linear.input_dimension
    out_dim = gt_linear.output_dimension

    # Transpose GoTorch (in, out) to PyTorch (out, in)
    w_data = list(gt_linear.linear.data.data)
    w_tensor = torch.tensor(w_data, dtype=torch.float32).reshape(in_dim, out_dim).t()
    pt_linear.weight.data.copy_(w_tensor)

    if gt_linear.bias is not None and pt_linear.bias is not None:
        b_data = list(gt_linear.bias.data.data)
        b_tensor = torch.tensor(b_data, dtype=torch.float32).reshape(out_dim)
        pt_linear.bias.data.copy_(b_tensor)


def sync_sequential_weights(gt_seq: Sequential, pt_seq: torch.nn.Sequential) -> None:
    """Synchronizes all Linear layers in GoTorch Sequential to PyTorch Sequential."""
    assert len(gt_seq) == len(pt_seq)
    for gt_layer, pt_layer in zip(gt_seq, pt_seq):
        if isinstance(gt_layer, Linear) and isinstance(pt_layer, torch.nn.Linear):
            sync_linear_weights(gt_layer, pt_layer)


class TestSequentialActivationParity:
    """Validates forward pass outputs and backward gradients of Sequential containers

    combining Linear layers with standard activations against torch.nn equivalents.
    """

    @pytest.mark.parametrize(
        "gt_act_factory,pt_act_factory",
        [
            (lambda: ReLU(), lambda: torch.nn.ReLU()),
            (lambda: LeakyReLU(alpha=0.1), lambda: torch.nn.LeakyReLU(negative_slope=0.1)),
            (lambda: Sigmoid(), lambda: torch.nn.Sigmoid()),
            (lambda: Tanh(), lambda: torch.nn.Tanh()),
        ],
        ids=["relu", "leaky_relu", "sigmoid", "tanh"],
    )
    def test_single_layer_activation_forward_parity(
        self,
        gt_act_factory: Callable[[], Module],
        pt_act_factory: Callable[[], torch.nn.Module],
    ) -> None:
        """Verifies forward pass parity for Linear + Activation against torch.nn.Sequential."""
        gt_linear = Linear(input_dimension=3, output_dimension=2, bias=True)
        gt_linear.linear = Tensor(NDArray([1.0, -2.0, 3.0, -1.0, 0.5, 2.0], shape=(3, 2)))
        gt_linear.bias = Tensor(NDArray([0.2, -0.5], shape=(2,)))
        gt_model = Sequential(gt_linear, gt_act_factory())

        pt_linear = torch.nn.Linear(in_features=3, out_features=2, bias=True)
        sync_linear_weights(gt_linear, pt_linear)
        pt_model = torch.nn.Sequential(pt_linear, pt_act_factory())

        # Test batch with positive, negative, and zero inputs
        x_data = [[-1.5, 0.5, 2.0], [0.0, -0.8, 1.2]]
        gt_x = Tensor(NDArray([-1.5, 0.5, 2.0, 0.0, -0.8, 1.2], shape=(2, 3)))
        pt_x = torch.tensor(x_data, dtype=torch.float32)

        gt_out = gt_model(gt_x)
        pt_out = pt_model(pt_x)

        assert gt_out.shape == tuple(pt_out.shape)
        assert list(gt_out.data.data) == pytest.approx(pt_out.flatten().tolist(), abs=1e-5)

    @pytest.mark.parametrize(
        "gt_act_factory,pt_act_factory",
        [
            (lambda: ReLU(), lambda: torch.nn.ReLU()),
            (lambda: LeakyReLU(alpha=0.1), lambda: torch.nn.LeakyReLU(negative_slope=0.1)),
            (lambda: Sigmoid(), lambda: torch.nn.Sigmoid()),
            (lambda: Tanh(), lambda: torch.nn.Tanh()),
        ],
        ids=["relu", "leaky_relu", "sigmoid", "tanh"],
    )
    def test_single_layer_activation_backward_parity(
        self,
        gt_act_factory: Callable[[], Module],
        pt_act_factory: Callable[[], torch.nn.Module],
    ) -> None:
        """Verifies backward gradient parity (x.grad, weight.grad, bias.grad) against PyTorch."""
        gt_linear = Linear(input_dimension=3, output_dimension=2, bias=True)
        gt_linear.linear = Tensor(NDArray([0.5, -1.2, 1.0, 0.8, -0.4, 0.6], shape=(3, 2)))
        gt_linear.bias = Tensor(NDArray([0.1, -0.2], shape=(2,)))
        gt_model = Sequential(gt_linear, gt_act_factory())

        pt_linear = torch.nn.Linear(in_features=3, out_features=2, bias=True)
        sync_linear_weights(gt_linear, pt_linear)
        pt_model = torch.nn.Sequential(pt_linear, pt_act_factory())

        x_data = [[-1.0, 2.0, 0.5], [1.5, -0.5, -1.0]]
        gt_x = Tensor(NDArray([-1.0, 2.0, 0.5, 1.5, -0.5, -1.0], shape=(2, 3)))
        pt_x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)

        gt_out = gt_model(gt_x)
        pt_out = pt_model(pt_x)

        gt_out.backward()
        pt_out.backward(torch.ones_like(pt_out))

        # Check input gradient
        assert gt_x.grad is not None
        assert list(gt_x.grad.data) == pytest.approx(pt_x.grad.flatten().tolist(), abs=1e-5)

        # Check weight gradient: GoTorch (in, out) matches PyTorch (out, in).t()
        assert gt_linear.linear.grad is not None
        assert list(gt_linear.linear.grad.data) == pytest.approx(
            pt_linear.weight.grad.t().flatten().tolist(), abs=1e-5
        )

        # Check bias gradient
        assert gt_linear.bias.grad is not None
        assert list(gt_linear.bias.grad.data) == pytest.approx(
            pt_linear.bias.grad.flatten().tolist(), abs=1e-5
        )


class TestDeepSequentialParity:
    """Validates deep multi-layer Sequential architectures with mixed activations."""

    def test_deep_sequential_forward_and_backward_parity(self) -> None:
        """Verifies forward outputs and all layer gradients for a 4-layer MLP against PyTorch."""
        # 4-layer MLP: Linear(4, 6) -> ReLU -> Linear(6, 5) -> LeakyReLU -> Linear(5, 3) -> Tanh -> Linear(3, 2) -> Sigmoid
        gt_l1 = Linear(4, 6, std=0.2, bias=True)
        gt_l2 = Linear(6, 5, std=0.2, bias=True)
        gt_l3 = Linear(5, 3, std=0.2, bias=True)
        gt_l4 = Linear(3, 2, std=0.2, bias=True)
        gt_model = Sequential(
            gt_l1,
            ReLU(),
            gt_l2,
            LeakyReLU(alpha=0.05),
            gt_l3,
            Tanh(),
            gt_l4,
            Sigmoid(),
        )

        pt_l1 = torch.nn.Linear(4, 6, bias=True)
        pt_l2 = torch.nn.Linear(6, 5, bias=True)
        pt_l3 = torch.nn.Linear(5, 3, bias=True)
        pt_l4 = torch.nn.Linear(3, 2, bias=True)
        pt_model = torch.nn.Sequential(
            pt_l1,
            torch.nn.ReLU(),
            pt_l2,
            torch.nn.LeakyReLU(negative_slope=0.05),
            pt_l3,
            torch.nn.Tanh(),
            pt_l4,
            torch.nn.Sigmoid(),
        )

        # Sync weights across all linear layers
        sync_sequential_weights(gt_model, pt_model)

        # Input batch of 3 samples, feature dimension 4
        x_raw = [
            0.5, -1.2, 0.8, -0.3,
            -0.7, 0.4, 1.1, -0.9,
            0.2, -0.1, -0.6, 1.5,
        ]
        gt_x = Tensor(NDArray(x_raw, shape=(3, 4)))
        pt_x = torch.tensor(
            [
                [0.5, -1.2, 0.8, -0.3],
                [-0.7, 0.4, 1.1, -0.9],
                [0.2, -0.1, -0.6, 1.5],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )

        # 1. Forward pass parity
        gt_out = gt_model(gt_x)
        pt_out = pt_model(pt_x)

        assert gt_out.shape == (3, 2)
        assert list(gt_out.data.data) == pytest.approx(pt_out.flatten().tolist(), abs=1e-5)

        # 2. Backward pass parity
        gt_out.backward()
        pt_out.backward(torch.ones_like(pt_out))

        assert gt_x.grad is not None
        assert list(gt_x.grad.data) == pytest.approx(pt_x.grad.flatten().tolist(), abs=1e-5)

        # Verify gradients for all layers
        gt_linears = [gt_l1, gt_l2, gt_l3, gt_l4]
        pt_linears = [pt_l1, pt_l2, pt_l3, pt_l4]

        for i, (gt_l, pt_l) in enumerate(zip(gt_linears, pt_linears)):
            assert gt_l.linear.grad is not None, f"Layer {i} weight grad is None"
            assert list(gt_l.linear.grad.data) == pytest.approx(
                pt_l.weight.grad.t().flatten().tolist(), abs=1e-5
            ), f"Layer {i} weight gradient mismatch"

            assert gt_l.bias.grad is not None, f"Layer {i} bias grad is None"
            assert list(gt_l.bias.grad.data) == pytest.approx(
                pt_l.bias.grad.flatten().tolist(), abs=1e-5
            ), f"Layer {i} bias gradient mismatch"


class TestOptimizerParity:
    """Validates parameter optimization update parity against PyTorch optimizers."""

    def test_adam_step_parity_over_multiple_iterations(self) -> None:
        """Verifies GoTorch Adam parameter updates match PyTorch torch.optim.Adam over 3 steps."""
        gt_linear = Linear(input_dimension=2, output_dimension=2, bias=True)
        gt_linear.linear = Tensor(NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        gt_linear.bias = Tensor(NDArray([0.5, -0.5], shape=(2,)))
        gt_opt = Adam(gt_linear, lr=0.01, b1=0.9, b2=0.999, eps=1e-8)

        pt_linear = torch.nn.Linear(in_features=2, out_features=2, bias=True)
        sync_linear_weights(gt_linear, pt_linear)
        pt_opt = torch.optim.Adam(pt_linear.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        step_inputs = [
            [1.0, 2.0],
            [-1.5, 0.5],
            [0.8, -1.2],
        ]

        for step_idx, inp in enumerate(step_inputs):
            gt_opt.zero_grad()
            pt_opt.zero_grad()

            gt_x = Tensor(NDArray(inp, shape=(1, 2)))
            pt_x = torch.tensor([inp], dtype=torch.float32)

            gt_out = gt_linear(gt_x)
            pt_out = pt_linear(pt_x)

            gt_out.backward()
            pt_out.backward(torch.ones_like(pt_out))

            gt_opt.step()
            pt_opt.step()

            # Assert parameter weights and biases match exactly at each step
            assert list(gt_linear.linear.data.data) == pytest.approx(
                pt_linear.weight.data.t().flatten().tolist(), abs=1e-5
            ), f"Adam weight mismatch at step {step_idx + 1}"

            assert list(gt_linear.bias.data.data) == pytest.approx(
                pt_linear.bias.data.flatten().tolist(), abs=1e-5
            ), f"Adam bias mismatch at step {step_idx + 1}"

    def test_adam_deep_model_step_parity(self) -> None:
        """Verifies Adam optimization across a 2-layer MLP over multiple steps."""
        gt_l1 = Linear(3, 4, bias=True)
        gt_l2 = Linear(4, 2, bias=True)
        gt_model = Sequential(gt_l1, ReLU(), gt_l2)
        gt_opt = Adam(gt_model, lr=0.02, b1=0.9, b2=0.999, eps=1e-8)

        pt_l1 = torch.nn.Linear(3, 4, bias=True)
        pt_l2 = torch.nn.Linear(4, 2, bias=True)
        pt_model = torch.nn.Sequential(pt_l1, torch.nn.ReLU(), pt_l2)
        sync_sequential_weights(gt_model, pt_model)
        pt_opt = torch.optim.Adam(pt_model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-8)

        inputs = [
            [-0.5, 1.0, 2.0],
            [1.2, -0.4, 0.1],
            [0.3, 0.8, -1.5],
        ]

        for step_idx, inp in enumerate(inputs):
            gt_opt.zero_grad()
            pt_opt.zero_grad()

            gt_x = Tensor(NDArray(inp, shape=(1, 3)))
            pt_x = torch.tensor([inp], dtype=torch.float32)

            gt_out = gt_model(gt_x)
            pt_out = pt_model(pt_x)

            gt_out.backward()
            pt_out.backward(torch.ones_like(pt_out))

            gt_opt.step()
            pt_opt.step()

            # Verify layer 1
            assert list(gt_l1.linear.data.data) == pytest.approx(
                pt_l1.weight.data.t().flatten().tolist(), abs=1e-5
            ), f"MLP Layer 1 weight mismatch at step {step_idx + 1}"
            assert list(gt_l1.bias.data.data) == pytest.approx(
                pt_l1.bias.data.flatten().tolist(), abs=1e-5
            ), f"MLP Layer 1 bias mismatch at step {step_idx + 1}"

            # Verify layer 2
            assert list(gt_l2.linear.data.data) == pytest.approx(
                pt_l2.weight.data.t().flatten().tolist(), abs=1e-5
            ), f"MLP Layer 2 weight mismatch at step {step_idx + 1}"
            assert list(gt_l2.bias.data.data) == pytest.approx(
                pt_l2.bias.data.flatten().tolist(), abs=1e-5
            ), f"MLP Layer 2 bias mismatch at step {step_idx + 1}"

    def test_sgd_optimizer_step_parity(self) -> None:
        """Verifies standard SGD update step parity (theta = theta - lr * grad) against PyTorch SGD."""
        class SimpleSGD(Optimizer):
            """Minimal SGD optimizer implemented using the GoTorch Optimizer contract."""
            def __init__(self, module: Module, lr: float = 0.05) -> None:
                super().__init__(module, lr=lr, b1=0.0, b2=0.0)

            def step(self) -> None:
                for p in self.parameters:
                    if p.grad is not None:
                        new_data = [d - self.lr * g for d, g in zip(p.data.data, p.grad.data)]
                        p.data = NDArray(new_data, shape=p.data.shape)

        gt_linear = Linear(input_dimension=2, output_dimension=2, bias=True)
        gt_linear.linear = Tensor(NDArray([1.5, -0.5, 0.5, 2.0], shape=(2, 2)))
        gt_linear.bias = Tensor(NDArray([0.1, -0.3], shape=(2,)))
        gt_sgd = SimpleSGD(gt_linear, lr=0.05)

        pt_linear = torch.nn.Linear(in_features=2, out_features=2, bias=True)
        sync_linear_weights(gt_linear, pt_linear)
        pt_sgd = torch.optim.SGD(pt_linear.parameters(), lr=0.05, momentum=0.0)

        inputs = [
            [0.5, 1.5],
            [-1.0, 0.8],
            [1.2, -0.2],
        ]

        for step_idx, inp in enumerate(inputs):
            gt_sgd.zero_grad()
            pt_sgd.zero_grad()

            gt_x = Tensor(NDArray(inp, shape=(1, 2)))
            pt_x = torch.tensor([inp], dtype=torch.float32)

            gt_out = gt_linear(gt_x)
            pt_out = pt_linear(pt_x)

            gt_out.backward()
            pt_out.backward(torch.ones_like(pt_out))

            gt_sgd.step()
            pt_sgd.step()

            assert list(gt_linear.linear.data.data) == pytest.approx(
                pt_linear.weight.data.t().flatten().tolist(), abs=1e-5
            ), f"SGD weight mismatch at step {step_idx + 1}"
            assert list(gt_linear.bias.data.data) == pytest.approx(
                pt_linear.bias.data.flatten().tolist(), abs=1e-5
            ), f"SGD bias mismatch at step {step_idx + 1}"


class TestLossParity:
    """Validates GoTorch.nn MSELoss and BCELoss forward and backward passes against torch.nn equivalents."""

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_mse_loss_forward_and_backward_parity(self, reduction: str) -> None:
        """Verifies forward loss value and input gradient of MSELoss against torch.nn.MSELoss."""
        random.seed(401)
        shape = (3, 4)
        num_elem = 12
        raw_pred = [random.uniform(-3.0, 3.0) for _ in range(num_elem)]
        raw_target = [random.uniform(-3.0, 3.0) for _ in range(num_elem)]

        gt_x = Tensor(NDArray(raw_pred, shape=shape))
        gt_y = Tensor(NDArray(raw_target, shape=shape))
        gt_criterion = MSELoss(reduction=reduction)
        gt_loss = gt_criterion(gt_x, gt_y)

        pt_x = torch.tensor(raw_pred, dtype=torch.float32).reshape(shape).requires_grad_(True)
        pt_y = torch.tensor(raw_target, dtype=torch.float32).reshape(shape)
        pt_criterion = torch.nn.MSELoss(reduction=reduction)
        pt_loss = pt_criterion(pt_x, pt_y)

        # Forward parity
        if reduction == "none":
            assert gt_loss.shape == shape
            assert list(gt_loss.data.data) == pytest.approx(pt_loss.flatten().tolist(), abs=1e-5)
        else:
            assert gt_loss.shape == ()
            assert pytest.approx(gt_loss.data[()], rel=1e-5) == pt_loss.item()

        # Backward parity
        if reduction == "none":
            raw_grad = [random.uniform(0.5, 2.0) for _ in range(num_elem)]
            gt_out_grad = Tensor(NDArray(raw_grad, shape=shape))
            pt_out_grad = torch.tensor(raw_grad, dtype=torch.float32).reshape(shape)
            gt_loss.backward(out_grad=gt_out_grad)
            pt_loss.backward(pt_out_grad)
        else:
            gt_loss.backward()
            pt_loss.backward()

        assert gt_x.grad is not None
        assert list(gt_x.grad.data) == pytest.approx(pt_x.grad.flatten().tolist(), abs=1e-5)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_bce_loss_forward_and_backward_parity(self, reduction: str) -> None:
        """Verifies forward loss value and input gradient of BCELoss against torch.nn.BCELoss."""
        random.seed(402)
        shape = (2, 5)
        num_elem = 10
        raw_pred = [random.uniform(0.05, 0.95) for _ in range(num_elem)]
        raw_target = [random.choice([0.0, 1.0]) for _ in range(num_elem)]

        gt_x = Tensor(NDArray(raw_pred, shape=shape))
        gt_y = Tensor(NDArray(raw_target, shape=shape))
        gt_criterion = BCELoss(reduction=reduction)
        gt_loss = gt_criterion(gt_x, gt_y)

        pt_x = torch.tensor(raw_pred, dtype=torch.float32).reshape(shape).requires_grad_(True)
        pt_y = torch.tensor(raw_target, dtype=torch.float32).reshape(shape)
        pt_criterion = torch.nn.BCELoss(reduction=reduction)
        pt_loss = pt_criterion(pt_x, pt_y)

        # Forward parity
        if reduction == "none":
            assert gt_loss.shape == shape
            assert list(gt_loss.data.data) == pytest.approx(pt_loss.flatten().tolist(), abs=1e-5)
        else:
            assert gt_loss.shape == ()
            assert pytest.approx(gt_loss.data[()], rel=1e-5) == pt_loss.item()

        # Backward parity
        if reduction == "none":
            raw_grad = [random.uniform(0.5, 2.0) for _ in range(num_elem)]
            gt_out_grad = Tensor(NDArray(raw_grad, shape=shape))
            pt_out_grad = torch.tensor(raw_grad, dtype=torch.float32).reshape(shape)
            gt_loss.backward(out_grad=gt_out_grad)
            pt_loss.backward(pt_out_grad)
        else:
            gt_loss.backward()
            pt_loss.backward()

        assert gt_x.grad is not None
        assert list(gt_x.grad.data) == pytest.approx(pt_x.grad.flatten().tolist(), abs=1e-5)

    def test_loss_linear_training_step_parity(self) -> None:
        """Verifies end-to-end model parameter updates with MSELoss + Adam match PyTorch."""
        gt_linear = Linear(input_dimension=3, output_dimension=1, bias=True)
        gt_linear.linear = Tensor(NDArray([0.5, -0.2, 0.8], shape=(3, 1)))
        gt_linear.bias = Tensor(NDArray([0.1], shape=(1,)))
        gt_loss_fn = MSELoss(reduction="mean")
        gt_opt = Adam(gt_linear, lr=0.05, b1=0.9, b2=0.999)

        pt_linear = torch.nn.Linear(in_features=3, out_features=1, bias=True)
        sync_linear_weights(gt_linear, pt_linear)
        pt_loss_fn = torch.nn.MSELoss(reduction="mean")
        pt_opt = torch.optim.Adam(pt_linear.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-8)

        batches = [
            ([1.0, 0.5, -0.5, 0.0, 1.0, -1.0], [1.5, -0.5]),
            ([-0.5, 1.2, 0.3, 0.8, -0.4, 0.6], [0.2, 1.1]),
        ]

        for step_idx, (x_raw, y_raw) in enumerate(batches):
            gt_opt.zero_grad()
            pt_opt.zero_grad()

            gt_x = Tensor(NDArray(x_raw, shape=(2, 3)))
            gt_y = Tensor(NDArray(y_raw, shape=(2, 1)))
            pt_x = torch.tensor(x_raw, dtype=torch.float32).reshape(2, 3)
            pt_y = torch.tensor(y_raw, dtype=torch.float32).reshape(2, 1)

            gt_pred = gt_linear(gt_x)
            pt_pred = pt_linear(pt_x)

            gt_loss = gt_loss_fn(gt_pred, gt_y)
            pt_loss = pt_loss_fn(pt_pred, pt_y)

            assert pytest.approx(gt_loss.data[()], rel=1e-5) == pt_loss.item()

            gt_loss.backward()
            pt_loss.backward()

            gt_opt.step()
            pt_opt.step()

            assert list(gt_linear.linear.data.data) == pytest.approx(
                pt_linear.weight.data.t().flatten().tolist(), abs=1e-5
            ), f"Weight mismatch after step {step_idx + 1}"
            assert list(gt_linear.bias.data.data) == pytest.approx(
                pt_linear.bias.data.flatten().tolist(), abs=1e-5
            ), f"Bias mismatch after step {step_idx + 1}"

