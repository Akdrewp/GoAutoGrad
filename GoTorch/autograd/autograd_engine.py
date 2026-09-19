from __future__ import annotations

from typing import TYPE_CHECKING

from GoTorch.autograd.operations import Add
from GoTorch.backend.ndarray import NDArray

if TYPE_CHECKING:
    from GoTorch.tensor import Tensor


class AutogradEngine:
    """Orchestrates reverse-mode automatic differentiation over a DAG"""

    @classmethod
    def backward(cls, target: "Tensor", out_grad: "Tensor" = None) -> None:
        """Executes the backward pass starting from the target node.

        1. Create DFS array with no repeats
        2. Calculate gradient for this node
        3. For each node in DFS:
            4. Calculate parent gradient
            5. Add gradient to parent gradient property

        Args:
            target: The terminal node.
            out_grad: Optional external gradient.
            Defaults to a tensor of 1s with shape=self.data.shape
        """
        ordered_nodes = cls._topological_sort(target)
        cls._seed_gradient(target, out_grad)

        for node in reversed(ordered_nodes):
            if node.op is None:
                continue

            grad_tensor = (
                node.grad if isinstance(node.grad, Tensor) else Tensor(node.grad)
            )
            parent_grads = node.op.gradient(grad_tensor, node)
            cls._accumulate_parent_gradients(node, parent_grads)

    @classmethod
    def _topological_sort(cls, root: "Tensor") -> list["Tensor"]:
        """Orders graph nodes using post-order depth-first search.

        Args:
            root: The terminal node to trace backwards from.

        Returns:
            A list of nodes ordered from inputs/leaves to output.
        """
        topo: list["Tensor"] = []
        visited: set["Tensor"] = set()

        def build_topo(node: "Tensor") -> None:
            if node not in visited:
                visited.add(node)
                for parent in node.inputs:
                    build_topo(parent)
                topo.append(node)

        build_topo(root)
        return topo

    @classmethod
    def _seed_gradient(cls, target: "Tensor", out_grad: "Tensor" = None) -> None:
        """Initializes the seed gradient for the backward pass.

        Args:
            target: The terminal node receiving the initial gradient.
            out_grad: Optional external activation gradient. Defaults to tesnor of 1s.
        """
        if out_grad is not None:
            target.grad = out_grad.data if isinstance(out_grad, Tensor) else out_grad
        elif target.grad is None:
            target.grad = NDArray.ones_like(target.data)

    @classmethod
    def _accumulate_parent_gradients(
        cls,
        node: "Tensor",
        parent_grads: tuple["Tensor", ...],
    ) -> None:
        """Accumulates calculated gradients into parent input nodes.

        Args:
            node: The current node whose parents receive gradients.
            parent_grads: Tuple of gradients calculated for each parent input.
        """
        for parent, p_grad in zip(node.inputs, parent_grads):
            grad_data = p_grad.data if isinstance(p_grad, Tensor) else p_grad
            if parent.grad is None:
                parent.grad = grad_data
            else:
                parent.grad = Add.compute(parent.grad, grad_data)
