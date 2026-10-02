from __future__ import annotations

import math
import random
import pytest

from GoTorch.nn.activations import LeakyReLU, ReLU, Sigmoid, Tanh
from GoTorch.nn.layers import Linear
from GoTorch.nn.module import Sequential
from GoTorch.nn.optimizer import Adam
from GoTorch.tensor import Tensor


class TestFunctionApproximation:
    """Verifies non-linear function approximation across all four activations using Sequential and Adam."""

    def test_quadratic_approximation_with_relu(self, backend) -> None:
        """Approximates f(x) = x^2 over [-1.0, 1.0] using Linear -> ReLU -> Linear -> ReLU -> Linear."""
        random.seed(42)
        num_points = 11
        xs = [-1.0 + 2.0 * i / (num_points - 1) for i in range(num_points)]
        ys = [x_val ** 2 for x_val in xs]

        x_tensor = Tensor(backend.NDArray(xs, shape=(num_points, 1)))
        target_tensor = Tensor(backend.NDArray(ys, shape=(num_points, 1)))

        model = Sequential(
            Linear(1, 16, std=0.2, bias=True),
            ReLU(),
            Linear(16, 16, std=0.2, bias=True),
            ReLU(),
            Linear(16, 1, std=0.2, bias=True),
        )
        optimizer = Adam(model, lr=0.03, b1=0.9, b2=0.999)

        initial_pred = model(x_tensor)
        initial_diff = initial_pred - target_tensor
        initial_loss = (initial_diff.transpose(0, 1) @ initial_diff).data.data[0]

        for epoch in range(250):
            optimizer.zero_grad()
            pred = model(x_tensor)
            diff = pred - target_tensor
            loss = diff.transpose(0, 1) @ diff
            loss.backward()

            if epoch == 0:
                for param in model.parameters():
                    assert param.grad is not None

            optimizer.step()

        final_pred = model(x_tensor)
        final_diff = final_pred - target_tensor
        final_loss = (final_diff.transpose(0, 1) @ final_diff).data.data[0]

        # Verify > 95% loss reduction
        assert final_loss < initial_loss * 0.05


    def test_sine_approximation_with_leaky_relu(self, backend) -> None:
        """Approximates f(x) = sin(x) using Sequential(Linear, LeakyReLU, Linear)."""
        random.seed(42)
        num_points = 11
        xs = [-1.0 + 2.0 * i / (num_points - 1) for i in range(num_points)]
        ys = [math.sin(x_val) for x_val in xs]

        x_tensor = Tensor(backend.NDArray(xs, shape=(num_points, 1)))
        target_tensor = Tensor(backend.NDArray(ys, shape=(num_points, 1)))

        model = Sequential(
            Linear(1, 32, std=0.2, bias=True),
            LeakyReLU(alpha=0.1),
            Linear(32, 1, std=0.2, bias=True),
        )
        optimizer = Adam(model, lr=0.03, b1=0.9, b2=0.999)

        initial_pred = model(x_tensor)
        initial_diff = initial_pred - target_tensor
        initial_loss = (initial_diff.transpose(0, 1) @ initial_diff).data.data[0]

        for epoch in range(250):
            optimizer.zero_grad()
            pred = model(x_tensor)
            diff = pred - target_tensor
            loss = diff.transpose(0, 1) @ diff
            loss.backward()

            if epoch == 0:
                for param in model.parameters():
                    assert param.grad is not None

            optimizer.step()

        final_pred = model(x_tensor)
        final_diff = final_pred - target_tensor
        final_loss = (final_diff.transpose(0, 1) @ final_diff).data.data[0]

        assert final_loss < initial_loss * 0.10

        mean_abs_err = sum(abs(p - y) for p, y in zip(final_pred.data.data, ys)) / num_points
        assert mean_abs_err < 0.25

    def test_trigonometric_sum_with_sigmoid(self, backend) -> None:
        """Approximates f(x) = sin(x) + cos(x) using Sequential(Linear, Sigmoid, Linear)."""
        random.seed(42)
        num_points = 11
        xs = [-1.0 + 2.0 * i / (num_points - 1) for i in range(num_points)]
        ys = [math.sin(x_val) + math.cos(x_val) for x_val in xs]

        x_tensor = Tensor(backend.NDArray(xs, shape=(num_points, 1)))
        target_tensor = Tensor(backend.NDArray(ys, shape=(num_points, 1)))

        model = Sequential(
            Linear(1, 32, std=0.2, bias=True),
            Sigmoid(),
            Linear(32, 1, std=0.2, bias=True),
        )
        optimizer = Adam(model, lr=0.03, b1=0.9, b2=0.999)

        initial_pred = model(x_tensor)
        initial_diff = initial_pred - target_tensor
        initial_loss = (initial_diff.transpose(0, 1) @ initial_diff).data.data[0]

        for epoch in range(250):
            optimizer.zero_grad()
            pred = model(x_tensor)
            diff = pred - target_tensor
            loss = diff.transpose(0, 1) @ diff
            loss.backward()

            if epoch == 0:
                for param in model.parameters():
                    assert param.grad is not None

            optimizer.step()

        final_pred = model(x_tensor)
        final_diff = final_pred - target_tensor
        final_loss = (final_diff.transpose(0, 1) @ final_diff).data.data[0]

        assert final_loss < initial_loss * 0.10

        mean_abs_err = sum(abs(p - y) for p, y in zip(final_pred.data.data, ys)) / num_points
        assert mean_abs_err < 0.25

    def test_sine_approximation_with_tanh(self, backend) -> None:
        """Approximates f(x) = sin(x) using Sequential(Linear, Tanh, Linear)."""
        random.seed(42)
        num_points = 11
        xs = [-1.0 + 2.0 * i / (num_points - 1) for i in range(num_points)]
        ys = [math.sin(x_val) for x_val in xs]

        x_tensor = Tensor(backend.NDArray(xs, shape=(num_points, 1)))
        target_tensor = Tensor(backend.NDArray(ys, shape=(num_points, 1)))

        model = Sequential(
            Linear(1, 16, std=0.2, bias=True),
            Tanh(),
            Linear(16, 1, std=0.2, bias=True),
        )
        optimizer = Adam(model, lr=0.03, b1=0.9, b2=0.999)

        initial_pred = model(x_tensor)
        initial_diff = initial_pred - target_tensor
        initial_loss = (initial_diff.transpose(0, 1) @ initial_diff).data.data[0]

        for epoch in range(250):
            optimizer.zero_grad()
            pred = model(x_tensor)
            diff = pred - target_tensor
            loss = diff.transpose(0, 1) @ diff
            loss.backward()

            if epoch == 0:
                for param in model.parameters():
                    assert param.grad is not None

            optimizer.step()

        final_pred = model(x_tensor)
        final_diff = final_pred - target_tensor
        final_loss = (final_diff.transpose(0, 1) @ final_diff).data.data[0]

        assert final_loss < initial_loss * 0.10

        mean_abs_err = sum(abs(p - y) for p, y in zip(final_pred.data.data, ys)) / num_points
        assert mean_abs_err < 0.25
