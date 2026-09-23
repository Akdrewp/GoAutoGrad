"""The wrapper for neural network layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from GoTorch.tensor import Tensor


class Module:
    """Container for defining layers for tensors to pass through.

    Holds the weights and biases in layers through attributes and provides
    a unified interface for forward passes and parameter management.
    """

    def __init__(self) -> None:
        """Initializes internal module state."""
        pass

    def forward(self, *args, **kwargs):
        """Defines the forward pass layer orderings and computation.

        Should be overridden by all subclasses.

        Args:
            *args: Positional arguments for forward computation.
            **kwargs: Keyword arguments for forward computation.

        Returns:
            Computed output Tensor or tuple of Tensors.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Calls the forward method on input arguments.

        Args:
            *args: Positional arguments passed to forward().
            **kwargs: Keyword arguments passed to forward().

        Returns:
            The result of self.forward(*args, **kwargs).
        """
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        """Returns a list of all parameters (Tensors) in the module and submodules.

        Recursively traverses object attributes to collect Tensor parameters from
        this module and any child modules, deduplicating shared parameters.

        Returns:
            A list of unique Tensor parameters.
        """
        from GoTorch.tensor import Tensor

        params: list[Tensor] = []
        visited_ids: set[int] = set()

        def _collect(mod: Module) -> None:
            for _, value in mod.__dict__.items():
                if isinstance(value, Tensor):
                    if id(value) not in visited_ids:
                        visited_ids.add(id(value))
                        params.append(value)
                elif isinstance(value, Module):
                    _collect(value)

        _collect(self)
        return params
