from __future__ import annotations

import pytest

from GoTorch.autograd.operations import Add, MatMul, Sub, Transpose
from GoTorch.tensor import Tensor


class TestTensorDAG:
    """Verifies that operations build correct computational DAG structures."""

    def test_add_parents(self, backend) -> None:
        """Verifies addition node contains both parent tensors in its DAG inputs."""
        a = Tensor(backend.NDArray([1.0, 2.0], shape=(2,)))
        b = Tensor(backend.NDArray([3.0, 4.0], shape=(2,)))
        c = a + b

        assert c.op is Add
        assert len(c.inputs) == 2
        assert a in c.inputs
        assert b in c.inputs
        assert c.inputs[0] is a
        assert c.inputs[1] is b
        assert list(c.data.data) == [4.0, 6.0]

    def test_matmul_parents(self, backend) -> None:
        """Verifies matmul node contains both parent tensors in its DAG inputs."""
        a = Tensor(backend.NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        b = Tensor(backend.NDArray([2.0, 0.0, 1.0, 2.0], shape=(2, 2)))
        c = a @ b

        assert c.op is MatMul
        assert len(c.inputs) == 2
        assert a in c.inputs
        assert b in c.inputs
        assert c.inputs[0] is a
        assert c.inputs[1] is b
        assert c.shape == (2, 2)
        assert list(c.data.data) == [4.0, 4.0, 10.0, 8.0]

    def test_subtract_parents(self, backend) -> None:
        """Verifies subtraction node contains both parent tensors in its DAG inputs."""
        a = Tensor(backend.NDArray([5.0, 7.0], shape=(2,)))
        b = Tensor(backend.NDArray([2.0, 3.0], shape=(2,)))
        c = a - b

        assert c.op is Sub
        assert len(c.inputs) == 2
        assert a in c.inputs
        assert b in c.inputs
        assert c.inputs[0] is a
        assert c.inputs[1] is b
        assert list(c.data.data) == [3.0, 4.0]

    def test_transpose_parents(self, backend) -> None:
        """Verifies transpose node contains the source tensor in its DAG inputs."""
        a = Tensor(backend.NDArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(2, 3)))
        c = a.transpose(0, 1)

        assert c.op is Transpose
        assert len(c.inputs) == 1
        assert a in c.inputs
        assert c.inputs[0] is a
        assert c.shape == (3, 2)

    def test_combined_operations_dag(self, backend) -> None:
        """Verifies DAG hierarchy across a chain combining all four operations."""
        # Graph: out = ((a @ b).transpose() - c) + d
        a = Tensor(backend.NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        b = Tensor(backend.NDArray([2.0, 0.0, 1.0, 2.0], shape=(2, 2)))
        c = Tensor(backend.NDArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2)))
        d = Tensor(backend.NDArray([5.0, 5.0, 5.0, 5.0], shape=(2, 2)))

        # 1. MatMul: [a, b] -> mm
        mm = a @ b
        assert mm.op is MatMul
        assert mm.inputs == [a, b]
        assert a in mm.inputs and b in mm.inputs

        # 2. Transpose: [mm] -> tr
        tr = mm.transpose(0, 1)
        assert tr.op is Transpose
        assert tr.inputs == [mm]
        assert mm in tr.inputs

        # 3. Subtract: [tr, c] -> sub_node
        sub_node = tr - c
        assert sub_node.op is Sub
        assert sub_node.inputs == [tr, c]
        assert tr in sub_node.inputs and c in sub_node.inputs

        # 4. Add: [sub_node, d] -> out
        out = sub_node + d
        assert out.op is Add
        assert out.inputs == [sub_node, d]
        assert sub_node in out.inputs and d in out.inputs

        # Verify leaves have empty inputs and op is None
        for leaf in (a, b, c, d):
            assert leaf.op is None
            assert leaf.inputs == []
