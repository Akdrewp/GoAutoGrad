from typing import Sequence

class NDArray:
    """Multidimensional array backed by a flat contiguous list of floats."""
    data: list[float]
    shape: tuple[int, ...]
    offset: int
    strides: tuple[int, ...]
    def __init__(self, data: list[float] | Sequence[float], shape: tuple[int, ...], strides: tuple[int, ...] | None = None, offset: int = 0) -> None: ...
    def __getitem__(self, indices: tuple[int, ...] | int) -> float: ...
    def __setitem__(self, indices: tuple[int, ...] | int, value: float) -> None: ...
    def is_contiguous(self) -> bool:
        """Returns True if memory layout matches default row-major strides."""
    def to_contiguous(self) -> NDArray:
        """Returns a contiguous copy if strided/viewed, or self if already contiguous"""
    def transpose(self, dim0: int = 0, dim1: int = 1) -> NDArray:
        """Creates a zero-copy transposed view by swapping shape and strides."""
    def ones_like(self) -> NDArray:
        """Returns an array of ones with identical shape and standard strides."""
    def zeros_like(self) -> NDArray:
        """Returns an array of zeros with identical shape and standard strides."""
    def matmul(self, other: NDArray) -> NDArray:
        """Performs 2D matrix multiplication: (M, K) @ (K, N) -> (M, N)."""
    def add(self, other: NDArray) -> NDArray:
        """Element-wise addition with broadcasting (pad to the left with 1s)."""
    def __add__(self, other: object) -> NDArray: ...
    def __radd__(self, other: object) -> NDArray: ...
