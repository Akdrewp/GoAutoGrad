from __future__ import annotations

import random
import pytest
import torch


class TestNDArrayPyTorchValidation:
    """Validates GoTorch NDArray operations against official PyTorch behavior."""

    def test_random_float_tanh_validation(self, backend) -> None:
        """Compares NDArray.tanh() against torch.tanh() on randomly generated float inputs."""
        random.seed(42)
        shape = (5, 10)
        num_elements = shape[0] * shape[1]
        raw_data = [random.uniform(-5.0, 5.0) for _ in range(num_elements)]

        # Compute with GoTorch NDArray
        gotorch_array = backend.NDArray(data=raw_data, shape=shape)
        gotorch_result = gotorch_array.tanh()

        # Compute with PyTorch
        torch_tensor = torch.tensor(raw_data, dtype=torch.float32).reshape(shape)
        torch_result = torch.tanh(torch_tensor)

        # Validate shape and element-wise numerical parity
        assert gotorch_result.shape == shape
        torch_flattened = torch_result.flatten().tolist()
        for actual, expected in zip(gotorch_result.data, torch_flattened):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == expected

    def test_random_float_sigmoid_validation(self, backend) -> None:
        """Compares NDArray.sigmoid() against torch.sigmoid() on 3D randomized float inputs."""
        random.seed(1337)
        shape = (3, 4, 5)
        num_elements = shape[0] * shape[1] * shape[2]
        raw_data = [random.uniform(-10.0, 10.0) for _ in range(num_elements)]

        # Compute with GoTorch NDArray
        gotorch_array = backend.NDArray(data=raw_data, shape=shape)
        gotorch_result = gotorch_array.sigmoid()

        # Compute with PyTorch
        torch_tensor = torch.tensor(raw_data, dtype=torch.float32).reshape(shape)
        torch_result = torch.sigmoid(torch_tensor)

        # Validate shape and element-wise numerical parity across active & saturation zones
        assert gotorch_result.shape == shape
        torch_flattened = torch_result.flatten().tolist()
        for actual, expected in zip(gotorch_result.data, torch_flattened):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == expected

    def test_random_float_leaky_relu_backward_validation(self, backend) -> None:
        """Compares NDArray.leaky_relu_backward() against PyTorch autograd gradient on random inputs."""
        random.seed(999)
        shape = (4, 6)
        num_elements = shape[0] * shape[1]
        alpha = 0.05
        raw_data = [random.uniform(-6.0, 6.0) for _ in range(num_elements)]
        grad_data = [random.uniform(-3.0, 3.0) for _ in range(num_elements)]

        # Compute with GoTorch NDArray
        gotorch_x = backend.NDArray(data=raw_data, shape=shape)
        gotorch_grad = backend.NDArray(data=grad_data, shape=shape)
        gotorch_result = gotorch_x.leaky_relu_backward(grad_output=gotorch_grad, alpha=alpha)

        # Compute with PyTorch autograd
        torch_x = torch.tensor(raw_data, dtype=torch.float32).reshape(shape).requires_grad_(True)
        torch_grad = torch.tensor(grad_data, dtype=torch.float32).reshape(shape)
        torch_y = torch.nn.functional.leaky_relu(torch_x, negative_slope=alpha)
        torch_y.backward(torch_grad)

        assert gotorch_result.shape == shape
        torch_flattened = torch_x.grad.flatten().tolist()
        for actual, expected in zip(gotorch_result.data, torch_flattened):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == expected

