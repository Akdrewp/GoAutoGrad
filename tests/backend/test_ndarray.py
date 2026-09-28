from __future__ import annotations

import pytest


class TestNDArrayAdd:
    """Verifies element-wise addition and broadcasting behavior."""

    def test_add_identical_shape(self, backend) -> None:
        """Element-wise addition over identical 1D shapes."""
        a = backend.NDArray(data=[1.0, 2.0, 3.0], shape=(3,), strides=(1,))
        b = backend.NDArray(data=[4.0, 5.0, 6.0], shape=(3,), strides=(1,))
        c = a + b
        assert c.shape == (3,)
        assert list(c.data) == [5.0, 7.0, 9.0]

    def test_add_broadcast_pad_left(self, backend) -> None:
        """Tests broadcasting when second operand is padded to the left with 1s."""
        # (2, 3) + (3,) -> (2, 3) + (1, 3) -> (2, 3)
        a = backend.NDArray(data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))
        b = backend.NDArray(data=[10.0, 20.0, 30.0], shape=(3,))
        c = a + b
        assert c.shape == (2, 3)
        assert list(c.data) == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]

    def test_add_broadcast_pad_left_reverse(self, backend) -> None:
        """Tests broadcasting when first operand is padded to the left with 1s."""
        # (3,) + (2, 3) -> (1, 3) + (2, 3) -> (2, 3)
        a = backend.NDArray(data=[10.0, 20.0, 30.0], shape=(3,))
        b = backend.NDArray(data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3))
        c = a + b
        assert c.shape == (2, 3)
        assert list(c.data) == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]

    def test_add_broadcast_both_dimensions(self, backend) -> None:
        """Tests broadcasting across different dimensions: (2, 1) + (1, 3) -> (2, 3)."""
        a = backend.NDArray(data=[1.0, 2.0], shape=(2, 1))
        b = backend.NDArray(data=[10.0, 20.0, 30.0], shape=(1, 3))
        c = a + b
        assert c.shape == (2, 3)
        assert list(c.data) == [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]

    def test_add_broadcast_strided_transposed(self, backend) -> None:
        """Tests addition with non-contiguous transposed view and broadcasting."""
        # a is (2, 3) transposed to (3, 2)
        a = backend.NDArray(data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3)).transpose()
        b = backend.NDArray(data=[10.0, 20.0], shape=(2,))
        c = a + b
        assert c.shape == (3, 2)
        assert list(c.data) == [11.0, 24.0, 12.0, 25.0, 13.0, 26.0]

    def test_add_shape_mismatch_raises(self, backend) -> None:
        """Incompatible shapes should raise ValueError matching 'shape mismatch'."""
        a = backend.NDArray(data=[0.0] * 6, shape=(2, 3))
        b = backend.NDArray(data=[0.0] * 4, shape=(4,))
        with pytest.raises(ValueError, match="shape mismatch"):
            _ = a + b

    def test_add_unsupported_operand(self, backend) -> None:
        """Adding non-NDArray operand should raise TypeError."""
        a = backend.NDArray(data=[1.0, 2.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a + 42


class TestNDArrayRelu:
    """Verifies rectified linear unit (ReLU) activation behavior."""

    def test_relu_positive_values(self, backend) -> None:
        """Positive numbers should remain unchanged."""
        a = backend.NDArray(data=[1.0, 2.5, 3.0, 4.5], shape=(2, 2))
        res = a.relu()
        assert res.shape == (2, 2)
        assert list(res.data) == [1.0, 2.5, 3.0, 4.5]

    def test_relu_negative_values(self, backend) -> None:
        """Negative numbers should be zeroed out."""
        a = backend.NDArray(data=[-1.0, -2.5, -0.1, -100.0], shape=(2, 2))
        res = a.relu()
        assert res.shape == (2, 2)
        assert list(res.data) == [0.0, 0.0, 0.0, 0.0]

    def test_relu_zeros(self, backend) -> None:
        """Zeros should remain zero."""
        a = backend.NDArray(data=[0.0, 0.0, 0.0], shape=(3,))
        res = a.relu()
        assert res.shape == (3,)
        assert list(res.data) == [0.0, 0.0, 0.0]

    def test_relu_mixed_values_multidimensional(self, backend) -> None:
        """Mixed positive, negative, and zero values with 3D shape."""
        data = [
            -1.0, 2.0,
            0.0, -3.0,
            4.0, -5.0,
            6.0, -7.0,
            -8.0, 9.0,
            10.0, -11.0,
        ]
        a = backend.NDArray(data=data, shape=(2, 3, 2))
        res = a.relu()
        assert res.shape == (2, 3, 2)
        expected = [
            0.0, 2.0,
            0.0, 0.0,
            4.0, 0.0,
            6.0, 0.0,
            0.0, 9.0,
            10.0, 0.0,
        ]
        assert list(res.data) == expected

    def test_relu_strided_non_contiguous(self, backend) -> None:
        """Tests relu on a non-contiguous transposed view."""
        a = backend.NDArray(data=[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], shape=(2, 3)).transpose()
        assert not a.is_contiguous()
        res = a.relu()
        assert res.shape == (3, 2)
        assert res.is_contiguous()
        assert list(res.data) == [1.0, 0.0, 0.0, 5.0, 3.0, 0.0]

    def test_relu_preserves_shape(self, backend) -> None:
        """Ensures arbitrary shapes (e.g. 1D, 4D) are preserved after relu."""
        a_1d = backend.NDArray(data=[-1.0, 1.0], shape=(2,))
        assert a_1d.relu().shape == (2,)

        shape_4d = (1, 2, 1, 3)
        data_4d = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]
        a_4d = backend.NDArray(data=data_4d, shape=shape_4d)
        res_4d = a_4d.relu()
        assert res_4d.shape == shape_4d
        assert list(res_4d.data) == [0.0, 2.0, 0.0, 4.0, 0.0, 6.0]

