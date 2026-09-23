from __future__ import annotations

import pytest
import torch

from GoTorch.autograd.autograd_engine import AutogradEngine
from GoTorch.autograd.operations import Add, MatMul, Sub, Transpose
from GoTorch.backend.native_backend import NDArray
from GoTorch.tensor import Tensor


class TestTopologicalSort:
    """Verifies graph node ordering and deduplication in AutogradEngine."""

    def test_topological_sort_linear(self, backend) -> None:
        """Verifies post-order DFS on a linear chain A -> B -> C."""
        a = Tensor(backend.NDArray([1.0, 2.0], shape=(2,)))
        b = Tensor(backend.NDArray([3.0, 4.0], shape=(2,)))
        c = a + b
        d = c + a

        ordered = AutogradEngine._topological_sort(d)

        # Expected: inputs/leaves visited before consumers
        assert ordered.index(a) < ordered.index(c)
        assert ordered.index(b) < ordered.index(c)
        assert ordered.index(c) < ordered.index(d)
        assert len(ordered) == len(set(ordered))  # No duplicates

    def test_topological_sort_diamond(self, backend) -> None:
        """Verifies topological order on a diamond DAG where one node forks and merges."""
        a = Tensor(backend.NDArray([1.0, 2.0], shape=(2,)))
        b = Tensor(backend.NDArray([3.0, 4.0], shape=(2,)))
        left = a + b
        right = a - b
        out = left + right

        ordered = AutogradEngine._topological_sort(out)

        assert ordered[-1] is out
        assert ordered.index(a) < ordered.index(left)
        assert ordered.index(a) < ordered.index(right)
        assert ordered.index(b) < ordered.index(left)
        assert ordered.index(b) < ordered.index(right)
        assert ordered.index(left) < ordered.index(out)
        assert ordered.index(right) < ordered.index(out)
        assert len(ordered) == 5


class TestSeedGradient:
    """Verifies gradient initialization for root/target nodes."""

    def test_default_seed_gradient(self, backend) -> None:
        """Verifies target gradient defaults to ones with identical shape."""
        target = Tensor(backend.NDArray([2.0, 3.0, 4.0, 5.0], shape=(2, 2)))
        AutogradEngine._seed_gradient(target, out_grad=None)

        assert target.grad is not None
        assert target.grad.shape == (2, 2)
        assert list(target.grad.data) == [1.0, 1.0, 1.0, 1.0]

    def test_custom_seed_gradient(self, backend) -> None:
        """Verifies target gradient accepts custom external gradient."""
        target = Tensor(backend.NDArray([2.0, 3.0], shape=(2,)))
        custom_grad = Tensor(backend.NDArray([0.5, 2.5], shape=(2,)))
        AutogradEngine._seed_gradient(target, out_grad=custom_grad)

        assert list(target.grad.data) == [0.5, 2.5]


class TestAutogradBackwardOperations:
    """Verifies reverse-mode autodiff through individual operations against PyTorch."""

    def test_backward_add(self, backend) -> None:
        """Verifies gradient propagation through element-wise addition.

        ```mermaid
        graph TD
            A["a (2,)"] --> C["Add (c = a + b)"]
            B["b (2,)"] --> C
        ```
        """
        a = Tensor(backend.NDArray([1.0, 2.0], shape=(2,)))
        b = Tensor(backend.NDArray([3.0, 4.0], shape=(2,)))
        c = a + b
        c.backward()

        a_pt = torch.tensor([1.0, 2.0], requires_grad=True)
        b_pt = torch.tensor([3.0, 4.0], requires_grad=True)
        c_pt = a_pt + b_pt
        c_pt.backward(torch.ones_like(c_pt))

        assert list(c.data.data) == pytest.approx(c_pt.tolist())
        assert list(a.grad.data) == pytest.approx(a_pt.grad.tolist())
        assert list(b.grad.data) == pytest.approx(b_pt.grad.tolist())

    def test_backward_subtract(self, backend) -> None:
        """Verifies gradient propagation through element-wise subtraction.

        ```mermaid
        graph TD
            A["a (2,)"] --> C["Sub (c = a - b)"]
            B["b (2,)"] --> C
        ```
        """
        a = Tensor(backend.NDArray([5.0, 6.0], shape=(2,)))
        b = Tensor(backend.NDArray([2.0, 1.0], shape=(2,)))
        c = a - b
        c.backward()

        a_pt = torch.tensor([5.0, 6.0], requires_grad=True)
        b_pt = torch.tensor([2.0, 1.0], requires_grad=True)
        c_pt = a_pt - b_pt
        c_pt.backward(torch.ones_like(c_pt))

        assert list(c.data.data) == pytest.approx(c_pt.tolist())
        assert list(a.grad.data) == pytest.approx(a_pt.grad.tolist())
        assert list(b.grad.data) == pytest.approx(b_pt.grad.tolist())

    def test_backward_matmul(self, backend) -> None:
        """Verifies adjoint rules for matrix multiplication: dL/dA = G @ B.T, dL/dB = A.T @ G.

        ```mermaid
        graph TD
            A["a (2, 2)"] --> C["MatMul (out = a @ b)"]
            B["b (2, 2)"] --> C
        ```
        """
        a = Tensor(backend.NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        b = Tensor(backend.NDArray([2.0, 0.0, 1.0, 2.0], shape=(2, 2)))
        out = a @ b
        out.backward()

        a_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_pt = torch.tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
        out_pt = a_pt @ b_pt
        out_pt.backward(torch.ones_like(out_pt))

        assert list(out.data.data) == pytest.approx(out_pt.flatten().tolist())
        assert list(a.grad.data) == pytest.approx(a_pt.grad.flatten().tolist())
        assert list(b.grad.data) == pytest.approx(b_pt.grad.flatten().tolist())

    def test_backward_transpose(self, backend) -> None:
        """Verifies gradient propagation through matrix transposition.

        ```mermaid
        graph TD
            A["a (2, 3)"] --> T["Transpose (t = a.transpose)"]
        ```
        """
        a = Tensor(backend.NDArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3)))
        t = a.transpose(0, 1)
        grad = Tensor(backend.NDArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(3, 2)))
        t.backward(grad)

        a_pt = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        t_pt = a_pt.transpose(0, 1)
        grad_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        t_pt.backward(grad_pt)

        # Forward output verification
        t_elements = [t.data[i, j] for i in range(3) for j in range(2)]
        assert t_elements == pytest.approx(t_pt.contiguous().flatten().tolist())

        # Gradient verification
        a_grad_elements = [a.grad[i, j] for i in range(2) for j in range(3)]
        assert a.grad.shape == (2, 3)
        assert a_grad_elements == pytest.approx(a_pt.grad.flatten().tolist())


class TestGradientAccumulation:
    """Verifies gradient accumulation when tensors are reused across multiple DAG paths."""

    def test_accumulate_same_tensor_multiple_inputs(self, backend) -> None:
        """Verifies y = x + x results in grad_x = 1 + 1 = 2.

        ```mermaid
        graph TD
            X["x (2,)"] -->|input 0| Y["Add (y = x + x)"]
            X -->|input 1| Y
        ```
        """
        x = Tensor(backend.NDArray([3.0, -1.0], shape=(2,)))
        y = x + x
        y.backward()

        x_pt = torch.tensor([3.0, -1.0], requires_grad=True)
        y_pt = x_pt + x_pt
        y_pt.backward(torch.ones_like(y_pt))

        assert list(y.data.data) == pytest.approx(y_pt.tolist())
        assert list(x.grad.data) == pytest.approx(x_pt.grad.tolist())

    def test_accumulate_diamond_graph(self, backend) -> None:
        """Verifies gradient accumulation from two distinct paths merging at leaf.

        ```mermaid
        graph TD
            A["a (2, 2)"] --> B["MatMul (b = a @ w1)"]
            W1["w1 (2, 2)"] --> B
            A --> C["MatMul (c = a @ w2)"]
            W2["w2 (2, 2)"] --> C
            B --> OUT["Add (out = b + c)"]
            C --> OUT
        ```
        """
        a = Tensor(backend.NDArray([1.0, 0.0, 0.0, 1.0], shape=(2, 2)))
        w1 = Tensor(backend.NDArray([2.0, 1.0, 0.0, 2.0], shape=(2, 2)))
        w2 = Tensor(backend.NDArray([0.0, 3.0, 1.0, 1.0], shape=(2, 2)))

        out = (a @ w1) + (a @ w2)
        out.backward()

        a_pt = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        w1_pt = torch.tensor([[2.0, 1.0], [0.0, 2.0]], requires_grad=True)
        w2_pt = torch.tensor([[0.0, 3.0], [1.0, 1.0]], requires_grad=True)

        out_pt = (a_pt @ w1_pt) + (a_pt @ w2_pt)
        out_pt.backward(torch.ones_like(out_pt))

        assert list(out.data.data) == pytest.approx(out_pt.flatten().tolist())
        assert list(a.grad.data) == pytest.approx(a_pt.grad.flatten().tolist())
        assert list(w1.grad.data) == pytest.approx(w1_pt.grad.flatten().tolist())
        assert list(w2.grad.data) == pytest.approx(w2_pt.grad.flatten().tolist())


class TestMultiLayerBackward:
    """Verifies backpropagation through multi-step computational graphs against PyTorch."""

    def test_two_layer_chain(self, backend) -> None:
        """Verifies chain rule across two successive matrix multiplications and bias addition.

        ```mermaid
        graph TD
            X["x (1, 2)"] --> H["MatMul (h = x @ w1)"]
            W1["w1 (2, 2)"] --> H
            H --> Y["MatMul (y = h @ w2)"]
            W2["w2 (2, 1)"] --> Y
            Y --> OUT["Add (out = y + b)"]
            B["b (1, 1)"] --> OUT
        ```
        """
        x = Tensor(backend.NDArray([1.0, 2.0], shape=(1, 2)))
        w1 = Tensor(backend.NDArray([1.0, 0.0, 0.0, 2.0], shape=(2, 2)))
        w2 = Tensor(backend.NDArray([3.0, 1.0], shape=(2, 1)))
        b = Tensor(backend.NDArray([0.5], shape=(1, 1)))

        h = x @ w1        # shape (1, 2): [1.0, 4.0]
        y = h @ w2        # shape (1, 1): [1.0*3.0 + 4.0*1.0] = [7.0]
        out = y + b       # shape (1, 1): [7.5]

        out.backward()

        x_pt = torch.tensor([[1.0, 2.0]], requires_grad=True)
        w1_pt = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
        w2_pt = torch.tensor([[3.0], [1.0]], requires_grad=True)
        b_pt = torch.tensor([[0.5]], requires_grad=True)

        h_pt = x_pt @ w1_pt
        y_pt = h_pt @ w2_pt
        out_pt = y_pt + b_pt
        out_pt.backward(torch.ones_like(out_pt))

        assert list(out.data.data) == pytest.approx(out_pt.flatten().tolist())
        assert list(x.grad.data) == pytest.approx(x_pt.grad.flatten().tolist())
        assert list(w1.grad.data) == pytest.approx(w1_pt.grad.flatten().tolist())
        assert list(w2.grad.data) == pytest.approx(w2_pt.grad.flatten().tolist())
        assert list(b.grad.data) == pytest.approx(b_pt.grad.flatten().tolist())
