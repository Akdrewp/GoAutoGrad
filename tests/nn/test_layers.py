from __future__ import annotations

import pytest

from GoTorch.backend.native_backend import NDArray
from GoTorch.nn.layers import Linear
from GoTorch.nn.module import Module
from GoTorch.tensor import Tensor


class TestModuleContainer:
    """Verifies Module base class functionality and parameter collection."""

    def test_module_parameters(self) -> None:
        """Verifies recursive parameter extraction across nested modules."""
        class SimpleNet(Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = Linear(2, 3, bias=True)
                self.fc2 = Linear(3, 1, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.fc2(self.fc1(x))

        net = SimpleNet()
        params = net.parameters()

        # fc1 has weight + bias (2 params)
        # fc2 has weight only (1 param)
        # Total = 3 Tensor parameters
        assert len(params) == 3
        assert net.fc1.linear in params
        assert net.fc1.bias in params
        assert net.fc2.linear in params

    def test_module_call_invokes_forward(self) -> None:
        """Verifies __call__ delegates to forward."""
        class DummyModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.called = False

            def forward(self, x: Tensor) -> Tensor:
                self.called = True
                return x

        m = DummyModule()
        x = Tensor(NDArray([1.0], shape=(1,)))
        res = m(x)

        assert m.called is True
        assert res is x

    def test_module_forward_raises_not_implemented(self) -> None:
        """Verifies base Module raises NotImplementedError when forward is not defined."""
        m = Module()
        with pytest.raises(NotImplementedError):
            m()
