from __future__ import annotations

import pytest

from GoTorch.autograd.operations import Add, MatMul
from GoTorch.tensor import Tensor


class TestAddOperation:
    """Verifies forward addition properties and local gradients."""

    def test_adds_two_tensors(self, backend) -> None:
        """Element-wise forward addition over identical buffer layouts."""
        a_data = backend.NDArray(data=[1.0, 2.0, 3.0], shape=(3,), strides=(1,))
        b_data = backend.NDArray(data=[4.0, 5.0, 6.0], shape=(3,), strides=(1,))

        result = Add.compute(a_data, b_data)

        expected = [5.0, 7.0, 9.0]
        assert result.shape == (3,)
        assert list(result.data) == expected

    def test_gradient(self, backend) -> None:
        """Local adjoint rule for addition: d(A+B)/dA = I, d(A+B)/dB = I."""
        a = Tensor(backend.NDArray([1.0, 2.0], shape=(2,), strides=(1,)))
        b = Tensor(backend.NDArray([3.0, 4.0], shape=(2,), strides=(1,)))
        out = Tensor(backend.NDArray([4.0, 6.0], shape=(2,),
                                     strides=(1,)), op=Add, inputs=[a, b])

        out_grad = Tensor(backend.NDArray([1.0, 1.0], shape=(2,), strides=(1,)))
        grad_a, grad_b = Add.gradient(out_grad, out)

        assert list(grad_a.data.data) == [1.0, 1.0]
        assert list(grad_b.data.data) == [1.0, 1.0]


class TestTensorShapeMismatch:
    """Operations between buffers with incompatible shapes."""

    def test_raises_on_shape_mismatch(self, backend) -> None:
        """Should raise ValueError when non-broadcasted shapes are combined."""
        a_data = backend.NDArray(data=[0.0] * 6, shape=(2, 3), strides=(3, 1))
        b_data = backend.NDArray(data=[0.0] * 20, shape=(4, 5), strides=(5, 1))

        with pytest.raises(ValueError, match="shape mismatch"):
            Add.compute(a_data, b_data)


class TestMatMulOperation:
    """Verifies matrix multiplication forward pass and gradient"""

    def test_forward(self, backend) -> None:
        """Verify 2D matrix multiplication: (2, 3) @ (3, 2) -> (2, 2)."""
        # A = [[1, 2, 3],
        #      [4, 5, 6]]
        a_data = backend.NDArray(
            data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape=(2, 3),
            strides=(3, 1),
        )
        # B = [[7,  8],
        #      [9,  1],
        #      [2,  3]]
        b_data = backend.NDArray(
            data=[7.0, 8.0, 9.0, 1.0, 2.0, 3.0],
            shape=(3, 2),
            strides=(2, 1),
        )

        result = MatMul.compute(a_data, b_data)

        # Row 0: [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [31.0, 19.0]
        # Row 1: [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [85.0, 55.0]
        expected = [31.0, 19.0, 85.0, 55.0]

        assert result.shape == (2, 2)
        assert list(result.data) == expected

    def test_gradient(self, backend) -> None:
        """Verify adjoint rules: grad_A = out_grad @ B.T, grad_B = A.T @ out_grad."""
        a_arr = backend.NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2), strides=(2, 1))
        b_arr = backend.NDArray([2.0, 0.0, 1.0, 2.0], shape=(2, 2), strides=(2, 1))

        a = Tensor(a_arr)
        b = Tensor(b_arr)
        out = Tensor(backend.NDArray([4.0, 4.0, 10.0, 8.0],
                                    shape=(2, 2), strides=(2, 1)),
                                    op=MatMul, inputs=[a, b])

        # Incoming loss gradient dL/dOut = [[1, 1], [1, 1]]
        out_grad = Tensor(backend.NDArray([1.0, 1.0, 1.0, 1.0],
                                          shape=(2, 2),
                                          strides=(2, 1)))

        grad_a, grad_b = MatMul.gradient(out_grad, out)

        # B.T = [[2, 1], [0, 2]]
        # grad_A = out_grad @ B.T
        # = [[1, 1], [1, 1]] @ [[2, 1], [0, 2]]
        # = [[2, 3], [2, 3]]
        assert grad_a.data.shape == (2, 2)
        assert list(grad_a.data.data) == [2.0, 3.0, 2.0, 3.0]

        # A.T = [[1, 3], [2, 4]]
        # grad_B = A.T @ out_grad
        # = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]]
        # = [[4, 4], [6, 6]]
        assert grad_b.data.shape == (2, 2)
        assert list(grad_b.data.data) == [4.0, 4.0, 6.0, 6.0]
