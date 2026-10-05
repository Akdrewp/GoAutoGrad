import math
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


class TestNDArrayReluBackward:
    """Verifies ReLU backward gradient computation."""

    def test_relu_backward_missing_grad_output_raises(self, backend) -> None:
        """Omitting grad_output must raise TypeError."""
        a = backend.NDArray(data=[-2.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.relu_backward()

    def test_relu_backward_none_grad_output_raises(self, backend) -> None:
        """Passing grad_output=None must raise TypeError."""
        a = backend.NDArray(data=[-2.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.relu_backward(None)

    def test_relu_backward_explicit_grad(self, backend) -> None:
        """Gradient output is element-wise multiplied with ReLU derivative."""
        a = backend.NDArray(data=[-2.0, -0.5, 0.0, 1.0, 3.0], shape=(5,))
        grad = backend.NDArray(data=[10.0, 20.0, 30.0, 40.0, 50.0], shape=(5,))
        res = a.relu_backward(grad)
        assert res.shape == (5,)
        assert list(res.data) == [0.0, 0.0, 0.0, 40.0, 50.0]

    def test_relu_backward_strided_non_contiguous(self, backend) -> None:
        """Tests relu_backward on a transposed view."""
        a = backend.NDArray(data=[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], shape=(2, 3)).transpose()
        grad = backend.NDArray(data=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], shape=(3, 2))
        res = a.relu_backward(grad)
        assert res.shape == (3, 2)
        assert res.is_contiguous()
        # a transposed: [1.0, -4.0, -2.0, 5.0, 3.0, -6.0]
        assert list(res.data) == [10.0, 0.0, 0.0, 40.0, 50.0, 0.0]

    def test_relu_backward_shape_mismatch_raises(self, backend) -> None:
        """Mismatched grad_output shape should raise ValueError."""
        a = backend.NDArray(data=[1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        grad = backend.NDArray(data=[1.0, 2.0, 3.0], shape=(3,))
        with pytest.raises(ValueError):
            _ = a.relu_backward(grad)



class TestNDArrayTanh:
    """Verifies hyperbolic tangent (tanh) activation behavior."""

    def test_tanh_zero(self, backend) -> None:
        """tanh(0) should equal 0.0."""
        a = backend.NDArray(data=[0.0], shape=(1,))
        res = a.tanh()
        assert res.shape == (1,)
        assert res[0] == 0.0

    def test_tanh_values(self, backend) -> None:
        """Standard float values should match math.tanh within float precision."""
        vals = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
        a = backend.NDArray(data=vals, shape=(6,))
        res = a.tanh()
        assert res.shape == (6,)
        for actual, x in zip(res.data, vals):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == math.tanh(x)

    def test_tanh_odd_symmetry(self, backend) -> None:
        """tanh(-x) == -tanh(x)."""
        vals = [0.1, 0.5, 1.0, 2.5]
        neg_vals = [-x for x in vals]
        a = backend.NDArray(data=vals, shape=(4,))
        neg_a = backend.NDArray(data=neg_vals, shape=(4,))
        res = a.tanh()
        neg_res = neg_a.tanh()
        for pos, neg in zip(res.data, neg_res.data):
            assert pytest.approx(pos, rel=1e-5, abs=1e-6) == -neg

    def test_tanh_saturation(self, backend) -> None:
        """Extreme positive and negative numbers saturate to 1.0 and -1.0."""
        a = backend.NDArray(data=[-40.0, 40.0], shape=(2,))
        res = a.tanh()
        assert pytest.approx(res[0], rel=1e-5) == -1.0
        assert pytest.approx(res[1], rel=1e-5) == 1.0

    def test_tanh_preserves_shape(self, backend) -> None:
        """Arbitrary multidimensional shapes are preserved."""
        shape = (2, 3, 2)
        data = [0.1 * i for i in range(12)]
        a = backend.NDArray(data=data, shape=shape)
        res = a.tanh()
        assert res.shape == shape

    def test_tanh_strided_non_contiguous(self, backend) -> None:
        """Tests tanh on transposed strided views."""
        a = backend.NDArray(data=[0.0, 1.0, 2.0, -1.0, -2.0, 0.5], shape=(2, 3)).transpose()
        assert not a.is_contiguous()
        res = a.tanh()
        assert res.shape == (3, 2)
        assert res.is_contiguous()
        expected = [
            math.tanh(0.0), math.tanh(-1.0),
            math.tanh(1.0), math.tanh(-2.0),
            math.tanh(2.0), math.tanh(0.5)
        ]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp


class TestNDArrayTanhBackward:
    """Verifies tanh backward gradient computation."""

    def test_tanh_backward_missing_grad_output_raises(self, backend) -> None:
        """Omitting grad_output must raise TypeError."""
        a = backend.NDArray(data=[0.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.tanh_backward()

    def test_tanh_backward_none_grad_output_raises(self, backend) -> None:
        """Passing grad_output=None must raise TypeError."""
        a = backend.NDArray(data=[0.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.tanh_backward(None)

    def test_tanh_backward_explicit_grad(self, backend) -> None:
        """Upstream gradient is multiplied with (1 - tanh(x)^2)."""
        vals = [-1.0, 0.0, 1.0]
        grads = [2.0, 3.0, 4.0]
        a = backend.NDArray(data=vals, shape=(3,))
        g = backend.NDArray(data=grads, shape=(3,))
        res = a.tanh_backward(g)
        assert res.shape == (3,)
        for actual, x, dy in zip(res.data, vals, grads):
            t = math.tanh(x)
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == dy * (1.0 - t * t)

    def test_tanh_backward_strided_non_contiguous(self, backend) -> None:
        """Tanh backward on transposed view."""
        a = backend.NDArray(data=[0.0, 1.0, -1.0, 2.0], shape=(2, 2)).transpose()
        g = backend.NDArray(data=[1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        res = a.tanh_backward(g)
        assert res.shape == (2, 2)
        assert res.is_contiguous()
        # a transposed: [0.0, -1.0, 1.0, 2.0]
        at_vals = [0.0, -1.0, 1.0, 2.0]
        for actual, x, dy in zip(res.data, at_vals, [1.0, 2.0, 3.0, 4.0]):
            t = math.tanh(x)
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == dy * (1.0 - t * t)

    def test_tanh_backward_shape_mismatch_raises(self, backend) -> None:
        """Mismatched shapes should raise ValueError."""
        a = backend.NDArray(data=[1.0, 2.0], shape=(2,))
        g = backend.NDArray(data=[1.0, 2.0, 3.0], shape=(3,))
        with pytest.raises(ValueError):
            _ = a.tanh_backward(g)



class TestNDArraySigmoid:
    """Verifies sigmoid activation behavior."""

    def test_sigmoid_zero(self, backend) -> None:
        """sigmoid(0) == 0.5."""
        a = backend.NDArray(data=[0.0], shape=(1,))
        res = a.sigmoid()
        assert res.shape == (1,)
        assert pytest.approx(res[0], rel=1e-5) == 0.5

    def test_sigmoid_values(self, backend) -> None:
        """Evaluates sigmoid: 1 / (1 + exp(-x))."""
        vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
        a = backend.NDArray(data=vals, shape=(5,))
        res = a.sigmoid()
        assert res.shape == (5,)
        for actual, x in zip(res.data, vals):
            exp_val = 1.0 / (1.0 + math.exp(-x))
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp_val

    def test_sigmoid_symmetry(self, backend) -> None:
        """sigmoid(-x) == 1 - sigmoid(x)."""
        vals = [0.2, 0.7, 1.5, 3.0]
        neg_vals = [-x for x in vals]
        a = backend.NDArray(data=vals, shape=(4,))
        neg_a = backend.NDArray(data=neg_vals, shape=(4,))
        res = a.sigmoid()
        neg_res = neg_a.sigmoid()
        for pos, neg in zip(res.data, neg_res.data):
            assert pytest.approx(pos + neg, rel=1e-5, abs=1e-6) == 1.0

    def test_sigmoid_saturation(self, backend) -> None:
        """Extreme values saturate near 0.0 and 1.0."""
        a = backend.NDArray(data=[-50.0, 50.0], shape=(2,))
        res = a.sigmoid()
        assert pytest.approx(res[0], abs=1e-6) == 0.0
        assert pytest.approx(res[1], rel=1e-5) == 1.0

    def test_sigmoid_preserves_shape(self, backend) -> None:
        """Multidimensional shapes are preserved."""
        shape = (2, 2, 2)
        a = backend.NDArray(data=[0.5] * 8, shape=shape)
        res = a.sigmoid()
        assert res.shape == shape

    def test_sigmoid_strided_non_contiguous(self, backend) -> None:
        """Sigmoid on non-contiguous transposed view."""
        a = backend.NDArray(data=[0.0, 1.0, 2.0, -1.0, -2.0, 0.5], shape=(2, 3)).transpose()
        assert not a.is_contiguous()
        res = a.sigmoid()
        assert res.shape == (3, 2)
        assert res.is_contiguous()
        expected = [
            1.0 / (1.0 + math.exp(-0.0)),
            1.0 / (1.0 + math.exp(1.0)),
            1.0 / (1.0 + math.exp(-1.0)),
            1.0 / (1.0 + math.exp(2.0)),
            1.0 / (1.0 + math.exp(-2.0)),
            1.0 / (1.0 + math.exp(-0.5)),
        ]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp


class TestNDArraySigmoidBackward:
    """Verifies sigmoid backward gradient computation."""

    def test_sigmoid_backward_missing_grad_output_raises(self, backend) -> None:
        """Omitting grad_output must raise TypeError."""
        a = backend.NDArray(data=[0.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.sigmoid_backward()

    def test_sigmoid_backward_none_grad_output_raises(self, backend) -> None:
        """Passing grad_output=None must raise TypeError."""
        a = backend.NDArray(data=[0.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.sigmoid_backward(None)

    def test_sigmoid_backward_explicit_grad(self, backend) -> None:
        """Upstream grad multiplied with s * (1 - s)."""
        vals = [-2.0, 0.0, 2.0]
        grads = [1.5, 2.5, 3.5]
        a = backend.NDArray(data=vals, shape=(3,))
        g = backend.NDArray(data=grads, shape=(3,))
        res = a.sigmoid_backward(g)
        assert res.shape == (3,)
        for actual, x, dy in zip(res.data, vals, grads):
            s = 1.0 / (1.0 + math.exp(-x))
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == dy * s * (1.0 - s)

    def test_sigmoid_backward_strided_non_contiguous(self, backend) -> None:
        """Sigmoid backward on non-contiguous transposed view."""
        a = backend.NDArray(data=[0.0, 1.0, -1.0, 2.0], shape=(2, 2)).transpose()
        g = backend.NDArray(data=[1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        res = a.sigmoid_backward(g)
        assert res.shape == (2, 2)
        assert res.is_contiguous()
        at_vals = [0.0, -1.0, 1.0, 2.0]
        for actual, x, dy in zip(res.data, at_vals, [1.0, 2.0, 3.0, 4.0]):
            s = 1.0 / (1.0 + math.exp(-x))
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == dy * s * (1.0 - s)

    def test_sigmoid_backward_shape_mismatch_raises(self, backend) -> None:
        """Mismatched shapes should raise ValueError."""
        a = backend.NDArray(data=[1.0, 2.0], shape=(2,))
        g = backend.NDArray(data=[1.0, 2.0, 3.0], shape=(3,))
        with pytest.raises(ValueError):
            _ = a.sigmoid_backward(g)



class TestNDArrayLeakyRelu:
    """Verifies LeakyReLU activation behavior."""

    def test_leaky_relu_default_alpha(self, backend) -> None:
        """Default alpha=0.01: x if x > 0 else 0.01 * x."""
        vals = [-200.0, -10.0, 0.0, 5.0, 10.0]
        a = backend.NDArray(data=vals, shape=(5,))
        res = a.leaky_relu()
        assert res.shape == (5,)
        expected = [-2.0, -0.1, 0.0, 5.0, 10.0]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp

    def test_leaky_relu_custom_alpha(self, backend) -> None:
        """Custom alpha scales negative numbers accordingly."""
        vals = [-10.0, -5.0, 0.0, 4.0]
        a = backend.NDArray(data=vals, shape=(2, 2))
        res = a.leaky_relu(alpha=0.2)
        assert res.shape == (2, 2)
        expected = [-2.0, -1.0, 0.0, 4.0]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp

    def test_leaky_relu_preserves_shape(self, backend) -> None:
        """Arbitrary multidimensional shapes are preserved."""
        shape = (2, 3, 2)
        a = backend.NDArray(data=[-2.0] * 12, shape=shape)
        res = a.leaky_relu(alpha=0.1)
        assert res.shape == shape
        for actual in res.data:
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == -0.2

    def test_leaky_relu_strided_non_contiguous(self, backend) -> None:
        """LeakyReLU on transposed view."""
        a = backend.NDArray(data=[10.0, -20.0, 30.0, -40.0, 50.0, -60.0], shape=(2, 3)).transpose()
        assert not a.is_contiguous()
        res = a.leaky_relu(alpha=0.1)
        assert res.shape == (3, 2)
        assert res.is_contiguous()
        expected = [10.0, -4.0, -2.0, 50.0, 30.0, -6.0]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp


class TestNDArrayLeakyReluBackward:
    """Verifies LeakyReLU backward gradient computation."""

    def test_leaky_relu_backward_missing_grad_output_raises(self, backend) -> None:
        """Omitting grad_output must raise TypeError."""
        a = backend.NDArray(data=[-2.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.leaky_relu_backward()

    def test_leaky_relu_backward_none_grad_output_raises(self, backend) -> None:
        """Passing grad_output=None must raise TypeError."""
        a = backend.NDArray(data=[-2.0, 1.0], shape=(2,))
        with pytest.raises(TypeError):
            _ = a.leaky_relu_backward(None)

    def test_leaky_relu_backward_default_alpha(self, backend) -> None:
        """Default alpha=0.01: x > 0 -> grad, x <= 0 -> 0.01 * grad."""
        vals = [-2.0, -0.5, 0.0, 1.0, 3.0]
        grads = [10.0, 20.0, 30.0, 40.0, 50.0]
        a = backend.NDArray(data=vals, shape=(5,))
        g = backend.NDArray(data=grads, shape=(5,))
        res = a.leaky_relu_backward(g)
        assert res.shape == (5,)
        expected = [0.1, 0.2, 0.3, 40.0, 50.0]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp

    def test_leaky_relu_backward_custom_alpha(self, backend) -> None:
        """Custom alpha scales negative gradient contributions."""
        vals = [-2.0, 0.0, 3.0]
        grads = [10.0, 20.0, 30.0]
        a = backend.NDArray(data=vals, shape=(3,))
        g = backend.NDArray(data=grads, shape=(3,))
        res = a.leaky_relu_backward(g, alpha=0.25)
        assert res.shape == (3,)
        expected = [2.5, 5.0, 30.0]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp

    def test_leaky_relu_backward_strided_non_contiguous(self, backend) -> None:
        """LeakyReLU backward on transposed view."""
        a = backend.NDArray(data=[1.0, -2.0, 3.0, -4.0], shape=(2, 2)).transpose()
        g = backend.NDArray(data=[10.0, 20.0, 30.0, 40.0], shape=(2, 2))
        res = a.leaky_relu_backward(grad_output=g, alpha=0.1)
        assert res.shape == (2, 2)
        assert res.is_contiguous()
        # a_t: [1.0, 3.0, -2.0, -4.0]
        expected = [10.0, 20.0, 3.0, 4.0]
        for actual, exp in zip(res.data, expected):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == exp

    def test_leaky_relu_backward_shape_mismatch_raises(self, backend) -> None:
        """Mismatched shapes should raise ValueError."""
        a = backend.NDArray(data=[1.0, 2.0], shape=(2,))
        g = backend.NDArray(data=[1.0, 2.0, 3.0], shape=(3,))
        with pytest.raises(ValueError):
            _ = a.leaky_relu_backward(grad_output=g)


class TestNDArrayIsEmpty:
    """Verifies isEmpty method on NDArray."""

    def test_is_empty_default_constructed(self, backend) -> None:
        """Default constructed NDArray has uninitialized storage (isEmpty == True)."""
        a = backend.NDArray()
        assert a.isEmpty() is True

    def test_is_empty_initialized_empty_storage(self, backend) -> None:
        """Initialized empty storage with shape (0,) has isEmpty == True."""
        a = backend.NDArray(data=[], shape=(0,))
        assert a.isEmpty() is True

    def test_is_empty_initialized_non_empty_storage(self, backend) -> None:
        """NDArray with data has isEmpty == False."""
        a = backend.NDArray(data=[1.0, 2.0], shape=(2,))
        assert a.isEmpty() is False

    def test_is_empty_scalar(self, backend) -> None:
        """0-D scalar has 1 element and has isEmpty == False."""
        a = backend.NDArray(data=[42.0], shape=())
        assert a.isEmpty() is False


class TestNDArrayScalar:
    """Verifies 0-D scalar support with empty shape () and num_elements = 1."""

    def test_scalar_empty_shape_and_num_elements(self, backend) -> None:
        """0-D scalar has empty shape (), size 1, accessible via s[()] and s.data[0]."""
        s = backend.NDArray(data=[3.14], shape=())
        assert s.shape == ()
        assert s.size() == 1
        assert s[()] == pytest.approx(3.14, rel=1e-5)
        assert s.data[0] == pytest.approx(3.14, rel=1e-5)
        assert s.isEmpty() is False

        # Indexing with 1-D index raises IndexError (dimensional mismatch)
        with pytest.raises(IndexError):
            _ = s[0]

    def test_scalar_addition(self, backend) -> None:
        """Scalar-scalar addition produces a 0-D scalar."""
        s1 = backend.NDArray(data=[2.5], shape=())
        s2 = backend.NDArray(data=[3.5], shape=())
        res = s1 + s2
        assert res.shape == ()
        assert res.size() == 1
        assert res[()] == pytest.approx(6.0, rel=1e-5)
        assert res.data[0] == pytest.approx(6.0, rel=1e-5)

    def test_scalar_activations(self, backend) -> None:
        """Activations and backward operations work correctly on 0-D scalars."""
        s = backend.NDArray(data=[-2.0], shape=())
        assert s.relu()[()] == 0.0
        assert s.tanh()[()] == pytest.approx(math.tanh(-2.0), rel=1e-5)
        assert s.sigmoid()[()] == pytest.approx(1.0 / (1.0 + math.exp(2.0)), rel=1e-5)
        assert s.leaky_relu(alpha=0.1)[()] == pytest.approx(-0.2, rel=1e-5)

        grad = backend.NDArray(data=[5.0], shape=())
        assert s.relu_backward(grad)[()] == 0.0
        assert s.leaky_relu_backward(grad, alpha=0.1)[()] == pytest.approx(0.5, rel=1e-5)


class TestNDArrayBCELoss:
    """Verifies binary cross entropy loss (BCELoss) forward operations."""

    def test_bce_loss_default_reduction_mean(self, backend) -> None:
        """Default reduction is 'mean' and returns a 0-D scalar."""
        preds = [0.2, 0.7, 0.4, 0.8]
        targets = [0.0, 1.0, 0.0, 1.0]
        a = backend.NDArray(data=preds, shape=(4,))
        t = backend.NDArray(data=targets, shape=(4,))

        res_m = a.BCELoss(t)
        res_f = backend.BCELoss(a, t)

        assert res_m.shape == ()
        assert res_f.shape == ()
        assert res_m.size() == 1
        assert res_f.size() == 1

        expected = sum(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)) for p, y in zip(preds, targets)) / 4.0
        assert pytest.approx(res_m[()], rel=1e-5, abs=1e-6) == expected
        assert pytest.approx(res_f[()], rel=1e-5, abs=1e-6) == expected

    def test_bce_loss_reduction_sum(self, backend) -> None:
        """Reduction 'sum' computes total unreduced loss sum."""
        preds = [0.1, 0.6, 0.3, 0.9]
        targets = [0.0, 1.0, 1.0, 0.0]
        a = backend.NDArray(data=preds, shape=(4,))
        t = backend.NDArray(data=targets, shape=(4,))

        res_m = a.BCELoss(t, reduction="sum")
        res_f = backend.BCELoss(a, t, reduction="sum")

        assert res_m.shape == ()
        assert res_f.shape == ()
        expected = sum(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)) for p, y in zip(preds, targets))
        assert pytest.approx(res_m[()], rel=1e-5, abs=1e-6) == expected
        assert pytest.approx(res_f[()], rel=1e-5, abs=1e-6) == expected

    def test_bce_loss_reduction_none(self, backend) -> None:
        """Reduction 'none' preserves element-wise loss shape."""
        preds = [0.25, 0.5, 0.75]
        targets = [1.0, 0.0, 1.0]
        a = backend.NDArray(data=preds, shape=(3,))
        t = backend.NDArray(data=targets, shape=(3,))

        res_m = a.BCELoss(t, reduction="none")
        res_f = backend.BCELoss(a, t, reduction="none")

        assert res_m.shape == (3,)
        assert res_f.shape == (3,)
        expected = [-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)) for p, y in zip(preds, targets)]
        for act_m, act_f, exp in zip(res_m.data, res_f.data, expected):
            assert pytest.approx(act_m, rel=1e-5, abs=1e-6) == exp
            assert pytest.approx(act_f, rel=1e-5, abs=1e-6) == exp

    def test_bce_loss_broadcast_dimensions(self, backend) -> None:
        """Tests broadcasting between prediction and target: (2, 1) and (1, 3) -> (2, 3)."""
        preds = [0.2, 0.8]
        targets = [0.0, 0.5, 1.0]
        a = backend.NDArray(data=preds, shape=(2, 1))
        t = backend.NDArray(data=targets, shape=(1, 3))

        res_none = a.BCELoss(t, reduction="none")
        assert res_none.shape == (2, 3)

        res_mean = a.BCELoss(t, reduction="mean")
        assert res_mean.shape == ()

        res_sum = a.BCELoss(t, reduction="sum")
        assert res_sum.shape == ()

        expected_elements = []
        for p in preds:
            for y in targets:
                expected_elements.append(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))

        for act, exp in zip(res_none.data, expected_elements):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp
        assert pytest.approx(res_mean[()], rel=1e-5, abs=1e-6) == sum(expected_elements) / 6.0
        assert pytest.approx(res_sum[()], rel=1e-5, abs=1e-6) == sum(expected_elements)

    def test_bce_loss_strided_non_contiguous(self, backend) -> None:
        """Tests BCELoss on non-contiguous transposed input."""
        raw_preds = [0.1, 0.4, 0.7, 0.3, 0.6, 0.9]
        a = backend.NDArray(data=raw_preds, shape=(2, 3)).transpose()
        assert not a.is_contiguous()

        targets = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        t = backend.NDArray(data=targets, shape=(3, 2))

        res_m = a.BCELoss(t, reduction="mean")
        res_f = backend.BCELoss(a, t, reduction="mean")

        a_transposed_vals = [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]
        expected = sum(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)) for p, y in zip(a_transposed_vals, targets)) / 6.0

        assert pytest.approx(res_m[()], rel=1e-5, abs=1e-6) == expected
        assert pytest.approx(res_f[()], rel=1e-5, abs=1e-6) == expected

    def test_bce_loss_shape_mismatch_raises(self, backend) -> None:
        """Non-broadcastable shape mismatch must raise ValueError."""
        a = backend.NDArray(data=[0.2] * 6, shape=(2, 3))
        t = backend.NDArray(data=[0.5] * 4, shape=(4,))
        with pytest.raises(ValueError, match="shape mismatch"):
            _ = a.BCELoss(t)
        with pytest.raises(ValueError, match="shape mismatch"):
            _ = backend.BCELoss(a, t)


class TestNDArrayBCELossBackward:
    """Verifies binary cross entropy loss backward gradient computation."""

    def test_bce_loss_backward_default_grad_output_mean(self, backend) -> None:
        """Default grad_output with reduction='mean' multiplies by (1 / N)."""
        preds = [0.2, 0.7, 0.4, 0.8]
        targets = [0.0, 1.0, 0.0, 1.0]
        a = backend.NDArray(data=preds, shape=(4,))
        t = backend.NDArray(data=targets, shape=(4,))

        res_m = a.bce_loss_backward(t)
        res_f = backend.bce_loss_backward(a, t)

        assert res_m.shape == (4,)
        assert res_f.shape == (4,)

        expected = [((p - y) / (p * (1.0 - p))) / 4.0 for p, y in zip(preds, targets)]
        for act_m, act_f, exp in zip(res_m.data, res_f.data, expected):
            assert pytest.approx(act_m, rel=1e-5, abs=1e-6) == exp
            assert pytest.approx(act_f, rel=1e-5, abs=1e-6) == exp

    def test_bce_loss_backward_custom_grad_output_mean(self, backend) -> None:
        """Explicit scalar grad_output scales the mean gradient."""
        preds = [0.2, 0.5, 0.8]
        targets = [0.0, 1.0, 0.0]
        a = backend.NDArray(data=preds, shape=(3,))
        t = backend.NDArray(data=targets, shape=(3,))
        g = backend.NDArray(data=[3.0], shape=())

        res_m = a.bce_loss_backward(t, grad_output=g, reduction="mean")
        res_f = backend.bce_loss_backward(a, t, grad_output=g, reduction="mean")

        assert res_m.shape == (3,)
        assert res_f.shape == (3,)

        expected = [3.0 * ((p - y) / (p * (1.0 - p))) / 3.0 for p, y in zip(preds, targets)]
        for act_m, act_f, exp in zip(res_m.data, res_f.data, expected):
            assert pytest.approx(act_m, rel=1e-5, abs=1e-6) == exp
            assert pytest.approx(act_f, rel=1e-5, abs=1e-6) == exp

    def test_bce_loss_backward_reduction_sum(self, backend) -> None:
        """Reduction 'sum' computes unscaled gradient (or scaled by scalar grad_output)."""
        preds = [0.2, 0.7, 0.4]
        targets = [0.0, 1.0, 0.0]
        a = backend.NDArray(data=preds, shape=(3,))
        t = backend.NDArray(data=targets, shape=(3,))

        res_default = a.bce_loss_backward(t, reduction="sum")
        expected_default = [(p - y) / (p * (1.0 - p)) for p, y in zip(preds, targets)]
        for act, exp in zip(res_default.data, expected_default):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

        g = backend.NDArray(data=[0.5], shape=())
        res_scaled = a.bce_loss_backward(t, grad_output=g, reduction="sum")
        for act, exp in zip(res_scaled.data, expected_default):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == 0.5 * exp

    def test_bce_loss_backward_reduction_none(self, backend) -> None:
        """Reduction 'none' multiplies element-wise with grad_output."""
        preds = [0.2, 0.6, 0.8]
        targets = [0.0, 1.0, 0.0]
        grads = [2.0, 0.5, 1.5]
        a = backend.NDArray(data=preds, shape=(3,))
        t = backend.NDArray(data=targets, shape=(3,))
        g = backend.NDArray(data=grads, shape=(3,))

        res_m = a.bce_loss_backward(t, grad_output=g, reduction="none")
        res_f = backend.bce_loss_backward(a, t, grad_output=g, reduction="none")

        assert res_m.shape == (3,)
        assert res_f.shape == (3,)

        expected = [dy * (p - y) / (p * (1.0 - p)) for p, y, dy in zip(preds, targets, grads)]
        for act_m, act_f, exp in zip(res_m.data, res_f.data, expected):
            assert pytest.approx(act_m, rel=1e-5, abs=1e-6) == exp
            assert pytest.approx(act_f, rel=1e-5, abs=1e-6) == exp

    def test_bce_loss_backward_strided_non_contiguous(self, backend) -> None:
        """Tests bce_loss_backward on non-contiguous transposed input."""
        raw_preds = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
        a = backend.NDArray(data=raw_preds, shape=(2, 3)).transpose()
        assert not a.is_contiguous()

        targets = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        t = backend.NDArray(data=targets, shape=(3, 2))

        res = a.bce_loss_backward(t, reduction="mean")
        assert res.shape == (3, 2)
        assert res.is_contiguous()

        a_transposed_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        expected = [((p - y) / (p * (1.0 - p))) / 6.0 for p, y in zip(a_transposed_vals, targets)]
        for act, exp in zip(res.data, expected):
            assert pytest.approx(act, rel=1e-5, abs=1e-6) == exp

    def test_bce_loss_backward_shape_mismatch_raises(self, backend) -> None:
        """Incompatible prediction/target or grad_output shapes must raise ValueError."""
        a = backend.NDArray(data=[0.3] * 4, shape=(2, 2))
        t = backend.NDArray(data=[0.5] * 3, shape=(3,))
        with pytest.raises(ValueError, match="shape mismatch"):
            _ = a.bce_loss_backward(t)

        valid_t = backend.NDArray(data=[0.5] * 4, shape=(2, 2))
        bad_g = backend.NDArray(data=[1.0] * 3, shape=(3,))
        with pytest.raises(ValueError, match="shape mismatch"):
            _ = a.bce_loss_backward(valid_t, grad_output=bad_g, reduction="none")






