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
