#include "ndarray.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

/**
 * @brief Computes standard row-major (C-contiguous) strides for a given shape.
 */
template <typename T>
std::vector<size_t> ndarray<T>::default_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    std::vector<size_t> str(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        str[i] = str[i + 1] * shape[i + 1];
    }
    return str;
}

/**
 * @brief Creates an independent deep copy of another ndarray and its storage.
 */
template <typename T>
void ndarray<T>::CreateDeepCopy(const ndarray& other) {
    shape = other.shape;
    strides = other.strides;
    offset = other.offset;

    if (other.storage) {
        storage = std::make_shared<Storage<T>>(other.storage->data);
    } else {
        storage = nullptr;
    }
}

/**
 * @brief Allocates and initializes ndarray storage from a raw buffer and shape.
 */
template <typename T>
void ndarray<T>::CreateNDArray(
    const T* array,
    const std::vector<size_t>& shape_dims,
    const std::vector<size_t>& custom_strides,
    size_t off
) {
    shape = shape_dims;
    offset = off;

    size_t total_size = num_elements(shape);

    if (custom_strides.empty()) {
        strides = default_strides(shape);
    } else {
        strides = custom_strides;
    }

    if (array != nullptr && total_size > 0) {
        storage = std::make_shared<Storage<T>>(array, total_size);
    } else {
        storage = std::make_shared<Storage<T>>(total_size);
    }
}

/**
 * @brief Constructs an ndarray by copying elements from a std::vector.
 */
template <typename T>
ndarray<T>::ndarray(
    const std::vector<T>& vec,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    size_t offset
) : offset(0) {
    this->shape = shape;
    this->offset = offset;
    this->strides = strides.empty() ? default_strides(shape) : strides;
    this->storage = std::make_shared<Storage<T>>(vec);
}

/**
 * @brief Creates a zero-copy transposed view by swapping two dimensions and their strides.
 */
template <typename T>
ndarray<T> ndarray<T>::transpose(size_t dim0, size_t dim1) const {
    if (shape.size() < 2) {
        return *this;
    }
    if (dim0 >= shape.size() || dim1 >= shape.size()) {
        throw std::out_of_range("Transpose dimension out of range");
    }
    std::vector<size_t> new_shape = shape;
    std::vector<size_t> new_strides = strides;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return ndarray(storage, std::move(new_shape), std::move(new_strides), offset);
}

/**
 * @brief Maps multidimensional coordinate indices to a flat memory buffer offset.
 * 
 * 1. Validates coordinate count matches array rank
 * 2. Computes linear offset by summing base offset and coordinate-stride products
 * 
 * On coordinate count mismatch:
 * Throws std::out_of_range
 */
template <typename T>
size_t ndarray<T>::index_to_offset(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::out_of_range(
            "Index dimensional mismatch: expected " +
            std::to_string(shape.size()) + " dims, got " +
            std::to_string(indices.size())
        );
    }
    size_t off = offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        off += indices[i] * strides[i];
    }
    return off;
}

// Operation logic

namespace {

/**
 * @brief Advances a coordinate vector by one in row-major order.
 * 
 * 1. Iterates backwards from innermost dimension
 * 2. Increments coordinate; if within bounds, returns
 * 3. Resets coordinate to 0 and carries over to next outer dimension
 */
inline void advance_coordinate(std::vector<size_t>& coord, const std::vector<size_t>& shape) {
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        coord[d]++;
        if (coord[d] < shape[d]) {
            break;
        }
        coord[d] = 0;
    }
}

/**
 * @brief Pads a shape to the left with 1s to match the target rank.
 *
 * (3), target_rank=3 -> (1, 1, 3) 
 * No padding added if already target rank
 */
inline std::vector<size_t> pad_shape(const std::vector<size_t>& shape, size_t target_rank) {
    std::vector<size_t> padded(target_rank, 1);
    size_t pad = target_rank - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        padded[pad + i] = shape[i];
    }
    return padded;
}

/**
 * @brief Validates broadcast compatibility between two padded shapes and determines output shape.
 * 
 * 1. Allocates output shape vector of size max_ndim
 * 2. Compares corresponding dimensions of padded shapes
 * 3. Sets output dimension if equal or broadcasts if one dimension is 1
 * 4. Throws std::invalid_argument if dimensions are incompatible
 * 
 * On shape mismatch:
 * Throws std::invalid_argument
 */
inline std::vector<size_t> broadcast_shapes(
    const std::vector<size_t>& padded_a,
    const std::vector<size_t>& padded_b
) {
    size_t max_ndim = padded_a.size();
    std::vector<size_t> out_shape(max_ndim);

    // 2-3. Broadcastable if dimensions match or one of them is 1
    for (size_t i = 0; i < max_ndim; ++i) {
        size_t dim_a = padded_a[i];
        size_t dim_b = padded_b[i];
        if (dim_a == dim_b) {
            out_shape[i] = dim_a;
        } else if (dim_a == 1) {
            out_shape[i] = dim_b;
        } else if (dim_b == 1) {
            out_shape[i] = dim_a;
        } else {
            throw std::invalid_argument(
                "shape mismatch: cannot broadcast shapes"
            );
        }
    }
    return out_shape;
}

/**
 * @brief Computes effective broadcasted strides for an operand.
 * 
 * 1. Initializes stride vector of target_rank with 0s
 * 2. Computes left-padding offset from shape size and target_rank
 * 3. Maps non-broadcasted dimensions (size > 1) to original strides
 */
inline std::vector<size_t> compute_broadcast_strides(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    size_t target_rank
) {
    std::vector<size_t> eff_strides(target_rank, 0);
    size_t pad = target_rank - shape.size();

    // 3. Map non-broadcasted dimensions (size > 1) to original strides
    for (size_t i = 0; i < target_rank; ++i) {
        if (i >= pad) {
            size_t orig_dim = i - pad;
            if (shape[orig_dim] > 1) {
                eff_strides[i] = strides[orig_dim];
            }
        }
    }
    return eff_strides;
}

/**
 * @brief Allocates output buffer with total element count matching shape.
 */
template <typename T>
inline std::vector<T> allocate_output_buffer(const std::vector<size_t>& shape) {
    return std::vector<T>(ndarray<T>::num_elements(shape));
}

/**
 * @brief Accumulates element values from respective storage offsets with broadcasting.
 * 
 * 1. Verifies storage buffers for both operands are allocated
 * 2. Initializes coordinate tracking vector
 * 3. Computes buffer offsets using effective strides and stores element sums into output buffer
 * 4. Advances coordinates in row-major order
 * 
 * On unallocated storage:
 * Throws std::runtime_error
 */
template <typename T, typename BinaryOp>
inline void accumulate_broadcast(
    const std::shared_ptr<Storage<T>>& storage_a,
    size_t offset_a,
    const std::shared_ptr<Storage<T>>& storage_b,
    size_t offset_b,
    const std::vector<size_t>& eff_strides_a,
    const std::vector<size_t>& eff_strides_b,
    const std::vector<size_t>& out_shape,
    std::vector<T>& out_vec,
    BinaryOp op
) {
    size_t total_elements = out_vec.size();
    if (total_elements == 0) {
        return;
    }
    if (!storage_a || !storage_b) {
        throw std::runtime_error("NDArray storage is unallocated");
    }
    const T* a_ptr = storage_a->data.data() + offset_a;
    const T* b_ptr = storage_b->data.data() + offset_b;
    size_t max_ndim = out_shape.size();

    std::vector<size_t> coord(max_ndim, 0);
    for (size_t out_idx = 0; out_idx < total_elements; ++out_idx) {
        // 3. Compute strided offsets and apply binary operation
        size_t off_a = 0;
        size_t off_b = 0;
        for (size_t d = 0; d < max_ndim; ++d) {
            off_a += coord[d] * eff_strides_a[d];
            off_b += coord[d] * eff_strides_b[d];
        }
        out_vec[out_idx] = op(a_ptr[off_a], b_ptr[off_b]);

        advance_coordinate(coord, out_shape);
    }
}

/**
 * @brief Applies an element-wise binary operation across two ndarrays with broadcasting.
 * 
 * 1. Pads both input shapes to the left with 1s to match the maximum rank
 * 2. Validates broadcast dimension compatibility and determines output shape
 * 3. Computes effective strides for both operands (0 for broadcasted dimensions)
 * 4. Allocates output buffer and iterates through coordinates in row-major order
 * 5. Applies binary operation to element values from respective storage offsets
 * 
 * On shape mismatch:
 * Throws std::invalid_argument
 * On unallocated storage:
 * Throws std::runtime_error
 */
template <typename T, typename BinaryOp>
inline ndarray<T> broadcast_binary_op(
    const ndarray<T>& a,
    const ndarray<T>& b,
    BinaryOp op
) {
    size_t ndim_a = a.shape.size();
    size_t ndim_b = b.shape.size();
    size_t max_ndim = std::max(ndim_a, ndim_b);

    // 1. Pad shapes to the left with 1s so both have rank max_ndim
    std::vector<size_t> padded_a = pad_shape(a.shape, max_ndim);
    std::vector<size_t> padded_b = pad_shape(b.shape, max_ndim);

    // 2. Validate broadcast compatibility and determine output shape
    std::vector<size_t> out_shape = broadcast_shapes(padded_a, padded_b);

    // 3. Compute effective broadcasted strides for both operands
    std::vector<size_t> eff_strides_a = compute_broadcast_strides(a.shape, a.strides, max_ndim);
    std::vector<size_t> eff_strides_b = compute_broadcast_strides(b.shape, b.strides, max_ndim);

    // 4. Allocate output buffer
    std::vector<T> out_vec = allocate_output_buffer<T>(out_shape);

    // 5. Apply binary operation across strided coordinates
    accumulate_broadcast(a.storage, a.offset, b.storage, b.offset,
                         eff_strides_a, eff_strides_b, out_shape, out_vec, op);

    return ndarray<T>(out_vec, out_shape);
}

}  // namespace

/**
 * @brief Performs element-wise addition with broadcasting.
 */
template <typename T>
ndarray<T> ndarray<T>::add(const ndarray<T>& addend) const {
    return broadcast_binary_op(*this, addend, std::plus<T>{});
}

/**
 * @brief Performs element-wise subtraction with broadcasting.
 */
template <typename T>
ndarray<T> ndarray<T>::sub(const ndarray<T>& subtrahend) const {
    return broadcast_binary_op(*this, subtrahend, std::minus<T>{});
}

/**
 * @brief Performs 2D matrix multiplication over two ndarrays.
 * 
 * 1. Validates both operands are 2D arrays
 * 2. Checks inner dimensions match (K1 == K2)
 * 3. Allocates output buffer of size M x N
 * 4. Computes dot products across inner dimensions respecting strided layouts
 * 5. Returns resulting M x N ndarray
 * 
 * On non-2D input or inner dimension mismatch:
 * Throws std::invalid_argument
 */
template <typename T>
ndarray<T> ndarray<T>::matmul(const ndarray<T>& other) const {
    // 1. Validates both operands are 2D arrays
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument(
            "matmul requires 2D arrays, got " +
            std::to_string(shape.size()) + " and " +
            std::to_string(other.shape.size())
        );
    }
    size_t m = shape[0];
    size_t k1 = shape[1];
    size_t k2 = other.shape[0];
    size_t n = other.shape[1];
    // 2. Checks inner dimensions match (K1 == K2)
    if (k1 != k2) {
        throw std::invalid_argument(
            "shape mismatch: inner dimensions " +
            std::to_string(k1) + " and " +
            std::to_string(k2) + " must match"
        );
    }
    // 3. Allocates output buffer of size M x N
    std::vector<T> out_data(m * n, 0);
    // 4. Computes dot products across inner dimensions respecting strided layouts
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T dot_sum = 0;
            for (size_t k = 0; k < k1; ++k) {
                dot_sum += (*this)({i, k}) * other({k, j});
            }
            out_data[i * n + j] = dot_sum;
        }
    }
    // 5. Returns resulting M x N ndarray
    return ndarray<T>(out_data, {m, n});
}

// Explicit template instantiations
template struct Storage<float>;
template struct Storage<double>;
template struct Storage<int>;

template class ndarray<float>;
template class ndarray<double>;
template class ndarray<int>;
