from __future__ import annotations

import pytest

from GoTorch.nn.activations import ReLU, Sigmoid
from GoTorch.nn.layers import Linear
from GoTorch.nn.module import Module, Sequential
from GoTorch.tensor import Tensor


class TestSequentialInit:
    """Verifies Sequential container construction across multiple input signatures."""

    def test_init_varargs(self) -> None:
        """Verifies construction using variable arguments: Sequential(m1, m2, m3)."""
        l1 = Linear(2, 3)
        l2 = ReLU()
        l3 = Linear(3, 1)
        seq = Sequential(l1, l2, l3)

        assert isinstance(seq, Module)
        assert len(seq) == 3
        assert seq[0] is l1
        assert seq[1] is l2
        assert seq[2] is l3

    def test_init_list(self) -> None:
        """Verifies construction using a list: Sequential([m1, m2])."""
        layers = [Linear(4, 8), Sigmoid(), Linear(8, 2)]
        seq = Sequential(layers)

        assert len(seq) == 3
        assert seq[0] is layers[0]
        assert seq[1] is layers[1]
        assert seq[2] is layers[2]

    def test_init_tuple(self) -> None:
        """Verifies construction using a tuple: Sequential((m1, m2))."""
        layers = (Linear(3, 3), ReLU())
        seq = Sequential(layers)

        assert len(seq) == 2
        assert seq[0] is layers[0]
        assert seq[1] is layers[1]

    def test_init_empty(self) -> None:
        """Verifies default empty Sequential initialization."""
        seq = Sequential()
        assert len(seq) == 0
        assert seq.layers == []


class TestSequentialContainer:
    """Verifies sequence container behaviors: len, getitem, and iteration."""

    def test_len(self) -> None:
        """Verifies len() reflects the number of layers."""
        seq = Sequential(Linear(2, 2), ReLU(), Linear(2, 2), Sigmoid())
        assert len(seq) == 4

    def test_getitem_indexing(self) -> None:
        """Verifies positive and negative indexing."""
        l1 = Linear(2, 4)
        l2 = ReLU()
        l3 = Linear(4, 1)
        seq = Sequential(l1, l2, l3)

        assert seq[0] is l1
        assert seq[1] is l2
        assert seq[2] is l3
        assert seq[-1] is l3
        assert seq[-2] is l2

    def test_getitem_out_of_range(self) -> None:
        """Verifies IndexError is raised for out-of-bounds indices."""
        seq = Sequential(Linear(2, 2), ReLU())
        with pytest.raises(IndexError):
            _ = seq[5]
        with pytest.raises(IndexError):
            _ = seq[-5]

    def test_iteration(self) -> None:
        """Verifies iteration yields layers in forward sequence order."""
        l1 = Linear(2, 3)
        l2 = ReLU()
        l3 = Linear(3, 1)
        seq = Sequential(l1, l2, l3)

        yielded = list(seq)
        assert yielded == [l1, l2, l3]


class TestSequentialForward:
    """Verifies forward execution and chaining through Sequential layers."""

    def test_forward_chaining(self, backend) -> None:
        """Verifies sequential forward pass y = W2 @ ReLU(W1 @ x + b1)."""
        seq = Sequential(
            Linear(2, 2, bias=True),
            ReLU(),
            Linear(2, 1, bias=False),
        )

        # Set deterministic weights for exact validation
        seq[0].linear = Tensor(backend.NDArray([1.0, -1.0, 2.0, -2.0], shape=(2, 2)))
        seq[0].bias = Tensor(backend.NDArray([0.5, 0.5], shape=(2,)))
        seq[2].linear = Tensor(backend.NDArray([2.0, 1.0], shape=(2, 1)))

        x = Tensor(backend.NDArray([1.0, 1.0], shape=(1, 2)))
        out = seq(x)

        # fc1: [1.0, 1.0] @ [[1.0, -1.0], [2.0, -2.0]] = [3.0, -3.0]
        # + b: [3.5, -2.5]
        # relu: [3.5, 0.0]
        # fc2: [3.5, 0.0] @ [[2.0], [1.0]] = 7.0
        assert out.shape == (1, 1)
        assert list(out.data.data) == pytest.approx([7.0])

    def test_forward_empty_passthrough(self, backend) -> None:
        """Empty Sequential returns input Tensor unchanged."""
        seq = Sequential()
        x = Tensor(backend.NDArray([1.0, 2.0], shape=(2,)))
        out = seq(x)
        assert out is x or list(out.data.data) == list(x.data.data)


class TestSequentialParameters:
    """Verifies recursive parameter collection in Sequential containers."""

    def test_parameters_collection(self) -> None:
        """Verifies parameters() aggregates all child layers while ignoring activations."""
        fc1 = Linear(2, 3, bias=True)
        act = ReLU()
        fc2 = Linear(3, 1, bias=False)
        seq = Sequential(fc1, act, fc2)

        params = seq.parameters()
        assert len(params) == 3
        assert fc1.linear in params
        assert fc1.bias in params
        assert fc2.linear in params

    def test_nested_sequential_parameters(self) -> None:
        """Verifies parameters() traverses nested Sequential modules recursively."""
        fc1 = Linear(2, 4, bias=True)
        fc2 = Linear(4, 2, bias=False)
        fc3 = Linear(2, 1, bias=True)

        inner_seq = Sequential(fc1, ReLU(), fc2)
        outer_seq = Sequential(inner_seq, Sigmoid(), fc3)

        params = outer_seq.parameters()
        # fc1 (2) + fc2 (1) + fc3 (2) = 5
        assert len(params) == 5
        assert fc1.linear in params
        assert fc1.bias in params
        assert fc2.linear in params
        assert fc3.linear in params
        assert fc3.bias in params


class TestSequentialBackward:
    """Verifies backpropagation through Sequential containers."""

    def test_backward_gradient_propagation(self, backend) -> None:
        """Verifies gradients backpropagate through all Sequential submodules to input."""
        seq = Sequential(
            Linear(2, 2, bias=True),
            ReLU(),
            Linear(2, 1, bias=False),
        )

        seq[0].linear = Tensor(backend.NDArray([1.0, 1.0, 1.0, 1.0], shape=(2, 2)))
        seq[0].bias = Tensor(backend.NDArray([0.0, 0.0], shape=(2,)))
        seq[2].linear = Tensor(backend.NDArray([2.0, 3.0], shape=(2, 1)))

        x = Tensor(backend.NDArray([1.0, 1.0], shape=(1, 2)))
        out = seq(x)

        grad_out = Tensor(backend.NDArray([1.0], shape=(1, 1)))
        out.backward(grad_out)

        # Gradients must accumulate on input and all constituent parameters
        assert x.grad is not None
        assert seq[0].linear.grad is not None
        assert seq[0].bias.grad is not None
        assert seq[2].linear.grad is not None
