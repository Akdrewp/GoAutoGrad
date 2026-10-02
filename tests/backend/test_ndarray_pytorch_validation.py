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

    def test_random_float_add_and_broadcasting_validation(self, backend) -> None:
        """Validates element-wise addition and broadcasting against torch.add (+)."""
        random.seed(101)
        # 1. Same-shape addition: (3, 4) + (3, 4)
        shape_a = (3, 4)
        data_a = [random.uniform(-5.0, 5.0) for _ in range(12)]
        data_b = [random.uniform(-5.0, 5.0) for _ in range(12)]

        gt_a = backend.NDArray(data=data_a, shape=shape_a)
        gt_b = backend.NDArray(data=data_b, shape=shape_a)
        gt_res = gt_a.add(gt_b)

        t_a = torch.tensor(data_a, dtype=torch.float32).reshape(shape_a)
        t_b = torch.tensor(data_b, dtype=torch.float32).reshape(shape_a)
        t_res = t_a + t_b

        assert gt_res.shape == t_res.shape
        for act, exp in zip(gt_res.data, t_res.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

        # 2. Broadcasting addition: (2, 3) + (3,)
        shape_c = (2, 3)
        shape_d = (3,)
        data_c = [random.uniform(-5.0, 5.0) for _ in range(6)]
        data_d = [random.uniform(-5.0, 5.0) for _ in range(3)]

        gt_c = backend.NDArray(data=data_c, shape=shape_c)
        gt_d = backend.NDArray(data=data_d, shape=shape_d)
        gt_bcast = gt_c + gt_d

        t_c = torch.tensor(data_c, dtype=torch.float32).reshape(shape_c)
        t_d = torch.tensor(data_d, dtype=torch.float32).reshape(shape_d)
        t_bcast = t_c + t_d

        assert gt_bcast.shape == t_bcast.shape
        for act, exp in zip(gt_bcast.data, t_bcast.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_random_float_sub_and_broadcasting_validation(self, backend) -> None:
        """Validates element-wise subtraction and broadcasting against torch.sub (-)."""
        random.seed(102)
        # Broadcasting subtraction: (2, 1) - (1, 3) -> (2, 3)
        shape_a = (2, 1)
        shape_b = (1, 3)
        data_a = [random.uniform(-10.0, 10.0) for _ in range(2)]
        data_b = [random.uniform(-10.0, 10.0) for _ in range(3)]

        gt_a = backend.NDArray(data=data_a, shape=shape_a)
        gt_b = backend.NDArray(data=data_b, shape=shape_b)
        gt_res = gt_a.sub(gt_b)

        t_a = torch.tensor(data_a, dtype=torch.float32).reshape(shape_a)
        t_b = torch.tensor(data_b, dtype=torch.float32).reshape(shape_b)
        t_res = t_a - t_b

        assert gt_res.shape == t_res.shape
        for act, exp in zip(gt_res.data, t_res.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_random_float_matmul_validation(self, backend) -> None:
        """Validates 2D matrix multiplication against torch.matmul (@)."""
        random.seed(103)
        # (4, 6) @ (6, 5) -> (4, 5)
        shape_a = (4, 6)
        shape_b = (6, 5)
        data_a = [random.uniform(-3.0, 3.0) for _ in range(24)]
        data_b = [random.uniform(-3.0, 3.0) for _ in range(30)]

        gt_a = backend.NDArray(data=data_a, shape=shape_a)
        gt_b = backend.NDArray(data=data_b, shape=shape_b)
        gt_res = gt_a.matmul(gt_b)

        t_a = torch.tensor(data_a, dtype=torch.float32).reshape(shape_a)
        t_b = torch.tensor(data_b, dtype=torch.float32).reshape(shape_b)
        t_res = torch.matmul(t_a, t_b)

        assert gt_res.shape == (4, 5)
        for act, exp in zip(gt_res.data, t_res.flatten().tolist()):
            assert pytest.approx(act, rel=1e-4, abs=1e-5) == exp

    def test_random_float_relu_and_backward_validation(self, backend) -> None:
        """Validates NDArray.relu() and relu_backward() against PyTorch autograd."""
        random.seed(104)
        shape = (3, 4, 2)
        num_elements = 24
        raw_x = [random.uniform(-5.0, 5.0) for _ in range(num_elements)]
        raw_dy = [random.uniform(-2.0, 2.0) for _ in range(num_elements)]

        # Forward
        gt_x = backend.NDArray(data=raw_x, shape=shape)
        gt_y = gt_x.relu()

        t_x = torch.tensor(raw_x, dtype=torch.float32).reshape(shape).requires_grad_(True)
        t_y = torch.relu(t_x)

        assert gt_y.shape == shape
        for act, exp in zip(gt_y.data, t_y.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

        # Backward
        gt_dy = backend.NDArray(data=raw_dy, shape=shape)
        gt_dx = gt_x.relu_backward(gt_dy)

        t_dy = torch.tensor(raw_dy, dtype=torch.float32).reshape(shape)
        t_y.backward(t_dy)

        assert gt_dx.shape == shape
        for act, exp in zip(gt_dx.data, t_x.grad.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_random_float_tanh_backward_validation(self, backend) -> None:
        """Validates NDArray.tanh_backward() against PyTorch autograd."""
        random.seed(105)
        shape = (4, 5)
        num_elements = 20
        raw_x = [random.uniform(-4.0, 4.0) for _ in range(num_elements)]
        raw_dy = [random.uniform(-2.0, 2.0) for _ in range(num_elements)]

        gt_x = backend.NDArray(data=raw_x, shape=shape)
        gt_dy = backend.NDArray(data=raw_dy, shape=shape)
        gt_dx = gt_x.tanh_backward(gt_dy)

        t_x = torch.tensor(raw_x, dtype=torch.float32).reshape(shape).requires_grad_(True)
        t_dy = torch.tensor(raw_dy, dtype=torch.float32).reshape(shape)
        t_y = torch.tanh(t_x)
        t_y.backward(t_dy)

        assert gt_dx.shape == shape
        for act, exp in zip(gt_dx.data, t_x.grad.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_random_float_sigmoid_backward_validation(self, backend) -> None:
        """Validates NDArray.sigmoid_backward() against PyTorch autograd."""
        random.seed(106)
        shape = (3, 6)
        num_elements = 18
        raw_x = [random.uniform(-6.0, 6.0) for _ in range(num_elements)]
        raw_dy = [random.uniform(-2.0, 2.0) for _ in range(num_elements)]

        gt_x = backend.NDArray(data=raw_x, shape=shape)
        gt_dy = backend.NDArray(data=raw_dy, shape=shape)
        gt_dx = gt_x.sigmoid_backward(gt_dy)

        t_x = torch.tensor(raw_x, dtype=torch.float32).reshape(shape).requires_grad_(True)
        t_dy = torch.tensor(raw_dy, dtype=torch.float32).reshape(shape)
        t_y = torch.sigmoid(t_x)
        t_y.backward(t_dy)

        assert gt_dx.shape == shape
        for act, exp in zip(gt_dx.data, t_x.grad.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_random_float_leaky_relu_validation(self, backend) -> None:
        """Validates NDArray.leaky_relu() against torch.nn.functional.leaky_relu."""
        random.seed(107)
        shape = (4, 5)
        num_elements = 20
        alpha = 0.05
        raw_x = [random.uniform(-6.0, 6.0) for _ in range(num_elements)]

        gt_x = backend.NDArray(data=raw_x, shape=shape)
        gt_y = gt_x.leaky_relu(alpha=alpha)

        t_x = torch.tensor(raw_x, dtype=torch.float32).reshape(shape)
        t_y = torch.nn.functional.leaky_relu(t_x, negative_slope=alpha)

        assert gt_y.shape == shape
        for act, exp in zip(gt_y.data, t_y.flatten().tolist()):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_random_float_transpose_validation(self, backend) -> None:
        """Validates NDArray.transpose() against torch.transpose()."""
        random.seed(108)
        shape = (3, 5)
        raw_x = [random.uniform(-5.0, 5.0) for _ in range(15)]

        gt_x = backend.NDArray(data=raw_x, shape=shape)
        gt_t = gt_x.transpose(0, 1)

        t_x = torch.tensor(raw_x, dtype=torch.float32).reshape(shape)
        t_t = torch.transpose(t_x, 0, 1).contiguous()

        assert gt_t.shape == tuple(t_t.shape)
        for r in range(gt_t.shape[0]):
            for c in range(gt_t.shape[1]):
                assert pytest.approx(gt_t[r, c], rel=1e-5, abs=1e-6) == t_t[r, c].item()



