"""Outwarding facing interface for tensor data representation

Defines the information relating to a tensor and stores its
data.

Attributes:
    strides: The stride shape for the tensor array.
    data: The contiguous array level data.
    shape: The shape of the tensor being represented.
    offset: The memory level offset.
"""


from __future__ import annotations

from typing import Sequence


class NDArray:
    """Multidimensional array backed by a flat contiguous list of floats."""

    def __init__(
        self,
        data: list[float] | Sequence[float],
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        offset: int = 0,
    ) -> None:
        self.data: list[float] = list(data)
        self.shape: tuple[int, ...] = tuple(shape)
        self.offset: int = offset

        if strides is None:
            self.strides: tuple[int, ...] = self._default_strides(self.shape)
        else:
            self.strides = tuple(strides)

    @staticmethod
    def _default_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculates standard C-contiguous (row-major) strides."""
        if not shape:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    def _index_to_offset(self, indices: tuple[int, ...]) -> int:
        """Maps multidimensional indices (i_0, i_1, ...) to flat buffer offset."""
        if len(indices) != len(self.shape):
            error_msg = (
               f"Index dimensional mismatch: expected {len(self.shape)} dims,"
               f" got {len(indices)}"
            )
            raise IndexError(error_msg)
        offset = self.offset
        for idx, stride in zip(indices, self.strides):
            offset += idx * stride
        return offset

    def __getitem__(self, indices: tuple[int, ...] | int) -> float:
        if isinstance(indices, int):
            indices = (indices,)
        return self.data[self._index_to_offset(indices)]

    def __setitem__(self, indices: tuple[int, ...] | int, value: float) -> None:
        if isinstance(indices, int):
            indices = (indices,)
        self.data[self._index_to_offset(indices)] = value

    def is_contiguous(self) -> bool:
        """Returns True if memory layout matches default row-major strides."""
        return self.strides == self._default_strides(self.shape) and self.offset == 0

    def to_contiguous(self) -> NDArray:
        """Returns a contiguous copy if strided/viewed, or self if already contiguous"""
        if self.is_contiguous():
            return self

        # Materialize non-contiguous views into standard order
        new_data: list[float] = []

        def _traverse(dim: int, current_idx: list[int]) -> None:
            if dim == len(self.shape):
                new_data.append(self[tuple(current_idx)])
                return
            for i in range(self.shape[dim]):
                current_idx.append(i)
                _traverse(dim + 1, current_idx)
                current_idx.pop()

        _traverse(0, [])
        return NDArray(new_data, self.shape)

    def transpose(self, dim0: int = 0, dim1: int = 1) -> NDArray:
        """Creates a zero-copy transposed view by swapping shape and strides."""
        if len(self.shape) < 2:
            return self

        new_shape = list(self.shape)
        new_strides = list(self.strides)

        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

        # Shares the exact same self.data pointer/reference
        return NDArray(
            data=self.data,
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            offset=self.offset,
        )

    def ones_like(self) -> NDArray:
        """Returns an array of ones with identical shape and standard strides."""
        size = 1
        for dim in self.shape:
            size *= dim
        return NDArray([1.0] * size, self.shape)

    def zeros_like(self) -> NDArray:
        """Returns an array of zeros with identical shape and standard strides."""
        size = 1
        for dim in self.shape:
            size *= dim
        return NDArray([0.0] * size, self.shape)

    def matmul(self, other: NDArray) -> NDArray:
        """Performs 2D matrix multiplication: (M, K) @ (K, N) -> (M, N)."""
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError(
                    f"matmul requires 2D arrays, got {self.shape} and {other.shape}"
            )

        m, k1 = self.shape
        k2, n = other.shape

        if k1 != k2:
            raise ValueError(
                f"shape mismatch: inner dimensions {k1} and {k2} must match"
            )

        # Allocate flat memory for resulting (M, N) matrix
        out_data = [0.0] * (m * n)
        result = NDArray(data=out_data, shape=(m, n))

        # Dot product: C[i, j] = sum_k (A[i, k] * B[k, j])
        for i in range(m):
            for j in range(n):
                dot_sum = 0.0
                for k in range(k1):
                    dot_sum += self[i, k] * other[k, j]
                result[i, j] = dot_sum

        return result

    def add(self, other: NDArray) -> NDArray:
        """Element-wise addition with broadcasting (pad to the left with 1s)."""
        ndim_a = len(self.shape)
        ndim_b = len(other.shape)
        max_ndim = max(ndim_a, ndim_b)

        padded_a = (1,) * (max_ndim - ndim_a) + self.shape
        padded_b = (1,) * (max_ndim - ndim_b) + other.shape

        out_shape: list[int] = []
        for da, db in zip(padded_a, padded_b):
            if da == db:
                out_shape.append(da)
            elif da == 1:
                out_shape.append(db)
            elif db == 1:
                out_shape.append(da)
            else:
                raise ValueError(
                    f"shape mismatch: cannot broadcast dimensions {da} and {db}"
                )

        eff_strides_a: list[int] = [0] * max_ndim
        eff_strides_b: list[int] = [0] * max_ndim

        pad_a = max_ndim - ndim_a
        for i in range(pad_a, max_ndim):
            orig_d = i - pad_a
            if self.shape[orig_d] > 1:
                eff_strides_a[i] = self.strides[orig_d]

        pad_b = max_ndim - ndim_b
        for i in range(pad_b, max_ndim):
            orig_d = i - pad_b
            if other.shape[orig_d] > 1:
                eff_strides_b[i] = other.strides[orig_d]

        total_elements = 1
        for d in out_shape:
            total_elements *= d
        if not out_shape:
            total_elements = 0

        out_data: list[float] = [0.0] * total_elements
        if total_elements > 0:
            coord = [0] * max_ndim
            for out_idx in range(total_elements):
                off_a = self.offset
                off_b = other.offset
                for d in range(max_ndim):
                    off_a += coord[d] * eff_strides_a[d]
                    off_b += coord[d] * eff_strides_b[d]
                out_data[out_idx] = self.data[off_a] + other.data[off_b]

                if max_ndim > 0:
                    for d in range(max_ndim - 1, -1, -1):
                        coord[d] += 1
                        if coord[d] < out_shape[d]:
                            break
                        coord[d] = 0

        return NDArray(out_data, tuple(out_shape))

    def __add__(self, other: object) -> NDArray:
        if not isinstance(other, NDArray):
            return NotImplemented
        return self.add(other)

    def __radd__(self, other: object) -> NDArray:
        if not isinstance(other, NDArray):
            return NotImplemented
        return other.add(self)
