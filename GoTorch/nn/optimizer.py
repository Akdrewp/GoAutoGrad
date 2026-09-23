"""Optimizers for updating module parameters during neural network training."""

from __future__ import annotations

import math
from typing import Sequence

from GoTorch.backend.native_backend import NDArray
from GoTorch.nn.module import Module
from GoTorch.tensor import Tensor


class Optimizer:
    """Base class for all neural network optimizers.

    Attributes:
        parameters: List of learnable Tensor parameters to optimize.
        lr: Learning rate.
        b1: First moment decay rate (or momentum factor).
        b2: Second moment decay rate.
    """

    def __init__(
        self,
        module: Module | Sequence[Tensor],
        lr: float,
        b1: float,
        b2: float,
    ) -> None:
        """Initializes class variables and registers parameters to optimize.

        Args:
            module: A Module instance or an iterable sequence of Tensor parameters.
            lr: Learning rate.
            b1: First moment decay rate.
            b2: Second moment decay rate.
        """
        if isinstance(module, Module):
            self.parameters: list[Tensor] = module.parameters()
        else:
            self.parameters = list(module)

        self.lr = lr
        self.b1 = b1
        self.b2 = b2

    def zero_grad(self) -> None:
        """Resets the gradients of all managed parameters to None."""
        for param in self.parameters:
            param.grad = None

    def step(self) -> None:
        """Performs a single optimization parameter update step.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Maintains exponential moving averages of past gradients (first moment)
    and past squared gradients (second moment).

    Attributes:
        parameters: List of learnable Tensor parameters to optimize.
        lr: Learning rate.
        b1: Exponential decay rate for first moment estimates.
        b2: Exponential decay rate for second moment estimates.
        eps: Small constant added to denominator for numerical stability.
        t: Integer step counter for bias corrections.
        m: Dictionary mapping parameter id to first moment estimates.
        v: Dictionary mapping parameter id to second moment estimates.
    """

    def __init__(
        self,
        module: Module | Sequence[Tensor],
        lr: float,
        b1: float,
        b2: float,
        eps: float = 1e-8,
    ) -> None:
        """Initializes the Adam optimizer and sets moments to 0.

        Args:
            module: A Module instance or sequence of Tensors to optimize.
            lr: Learning rate.
            b1: Exponential decay rate for first moment estimates.
            b2: Exponential decay rate for second moment estimates.
            eps: Term added to denominator for numerical stability. Defaults to 1e-8.
        """
        super().__init__(module, lr, b1, b2)
        self.eps = eps
        self.t = 0
        self.m: dict[int, list[float]] = {}
        self.v: dict[int, list[float]] = {}

        # Set first and second moments to 0 for all initial parameters
        for param in self.parameters:
            param_len = len(param.data.data)
            self.m[id(param)] = [0.0] * param_len
            self.v[id(param)] = [0.0] * param_len

    def step(self) -> None:
        """Executes one step of Adam optimization.

        1. Increments global optimization step counter.
        2. Iterates through managed parameters and initializes moment buffers if needed.
        3. Updates biased first and second moment moving averages using current gradients.
        4. Computes bias-corrected moments and calculates parameter delta updates.
        5. Applies updates to parameter data.
        """
        self.t += 1

        for param in self.parameters:
            if param.grad is None:
                continue

            param_data = list(param.data.data)
            grad_data = list(param.grad.data)
            param_len = len(param_data)
            pid = id(param)

            if pid not in self.m:
                self.m[pid] = [0.0] * param_len
                self.v[pid] = [0.0] * param_len

            m_vec = self.m[pid]
            v_vec = self.v[pid]

            # 3-7. Update moments, bias-correct, and compute delta updates
            delta_update = [0.0] * param_len
            bias_corr1 = 1.0 - (self.b1 ** self.t)
            bias_corr2 = 1.0 - (self.b2 ** self.t)

            for i in range(param_len):
                g = grad_data[i]
                m_vec[i] = self.b1 * m_vec[i] + (1.0 - self.b1) * g
                v_vec[i] = self.b2 * v_vec[i] + (1.0 - self.b2) * (g * g)

                m_hat = m_vec[i] / bias_corr1
                v_hat = v_vec[i] / bias_corr2

                delta_update[i] = (self.lr * m_hat) / (math.sqrt(v_hat) + self.eps)

            # 8. Apply updated buffer
            new_param_data = [p - d for p, d in zip(param_data, delta_update)]
            param.data = NDArray(new_param_data, shape=param.data.shape)
