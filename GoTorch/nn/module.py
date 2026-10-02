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

        def _collect(item: object) -> None:
            if isinstance(item, Tensor):
                if id(item) not in visited_ids:
                    visited_ids.add(id(item))
                    params.append(item)
            elif isinstance(item, Module):
                for _, value in item.__dict__.items():
                    _collect(value)
            elif isinstance(item, (list, tuple)):
                for elem in item:
                    _collect(elem)

        _collect(self)
        return params


class Sequential(Module):
    """A sequential container of Modules.

    Modules will be added to it in the order they are passed in the
    constructor. The forward() pass of `Sequential` chains the output of each
    submodule as the input to the next submodule.

    Attributes:
        layers: List of child Modules in execution order.
    """

    def __init__(self, *args: Module | list[Module] | tuple[Module, ...]) -> None:
        """Initializes the Sequential container.

        Args:
            *args: Either variable arguments of Modules, or a single list/tuple of Modules.
        """
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            self.layers: list[Module] = list(args[0])
        else:
            self.layers = list(args)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        """Sequentially applies each layer to the input.

        Args:
            x: Input Tensor to pass through the chain of layers.

        Returns:
            Output Tensor after passing sequentially through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self) -> int:
        """Returns the number of layers in the Sequential container."""
        return len(self.layers)

    def __getitem__(self, idx: int) -> Module:
        """Returns the layer at the given index."""
        return self.layers[idx]

    def __iter__(self):
        """Yields each layer in the Sequential container."""
        return iter(self.layers)

