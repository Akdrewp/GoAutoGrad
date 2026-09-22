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
