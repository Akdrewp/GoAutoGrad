from GoTorch.autograd.operations import Add as Add
from GoTorch.tensor import Tensor as Tensor

class AutogradEngine:
    """Orchestrates reverse-mode automatic differentiation over a DAG"""
    @classmethod
    def backward(cls, target: Tensor, out_grad: Tensor = None) -> None:
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
