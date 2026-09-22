#include "ndarray.hpp"

#include <algorithm>
#include <stdexcept>

/**
 * @brief Computes standard row-major (C-contiguous) strides for a given shape.
 * 
 * 1. Returns empty stride list if shape is empty
 * 2. Initializes stride vector of shape size with 1s
 * 3. Iterates backwards from second-to-last dimension, setting each stride
 *    to the product of next stride and dimension size
 */
template <typename T>
std::vector<size_t> ndarray<T>::default_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    std::vector<size_t> str(shape.size(), 1);
    // Last value in stride will always be 1
    // from second last value to first
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
 * 
 * 1. Assigns shape dimensions and memory offset
 * 2. Computes default strides if custom strides are omitted
 * 3. Allocates storage buffer (copies from pointer if provided, else allocates zero-filled)
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
 * 
 * 1. Returns self if shape rank is less than 2
 * 2. Validates dimension indices against array rank
 * 3. Swaps shape dimensions and corresponding strides
 * 4. Returns view sharing underlying storage buffer
 * 
 * On dimension out of bounds:
 * Throws std::out_of_range
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


/**
 * @brief Performs element-wise addition with broadcasting.
 * 
 * 1. Pads both input shapes to the left with 1s to match the maximum rank
 * 2. Validates broadcast dimension compatibility and determines output shape
 * 3. Computes effective strides for both operands (0 for broadcasted dimensions)
 * 4. Allocates output buffer and iterates through coordinates in row-major order
 * 5. Accumulates element values from respective storage offsets
 * 
 * On shape mismatch:
 * Throws std::invalid_argument
 * On unallocated storage:
 * Throws std::runtime_error
 */
template <typename T>
ndarray<T> ndarray<T>::add(const ndarray<T>& addend) const {
    size_t ndim_a = shape.size();
    size_t ndim_b = addend.shape.size();
    size_t max_ndim = std::max(ndim_a, ndim_b);

    // 1. Pad shapes to the left with 1s so both have rank max_ndim
    std::vector<size_t> padded_a(max_ndim, 1);
    std::vector<size_t> padded_b(max_ndim, 1);

    size_t pad_a = max_ndim - ndim_a;
    for (size_t i = 0; i < ndim_a; ++i) {
        padded_a[pad_a + i] = shape[i];
    }

    size_t pad_b = max_ndim - ndim_b;
    for (size_t i = 0; i < ndim_b; ++i) {
        padded_b[pad_b + i] = addend.shape[i];
    }

    // 2. Validate broadcast compatibility and determine output shape
    std::vector<size_t> out_shape(max_ndim);
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

    // 3. Compute effective broadcasted strides for both operands
    std::vector<size_t> eff_strides_a(max_ndim, 0);
    for (size_t i = 0; i < max_ndim; ++i) {
        if (i >= pad_a) {
            size_t orig_dim = i - pad_a;
            if (shape[orig_dim] > 1) {
                eff_strides_a[i] = strides[orig_dim];
            }
        }
    }

    std::vector<size_t> eff_strides_b(max_ndim, 0);
    for (size_t i = 0; i < max_ndim; ++i) {
        if (i >= pad_b) {
            size_t orig_dim = i - pad_b;
            if (addend.shape[orig_dim] > 1) {
                eff_strides_b[i] = addend.strides[orig_dim];
            }
        }
    }

    // 4. Allocate output buffer and iterate through coordinates in row-major order
    size_t total_elements = num_elements(out_shape);
    std::vector<T> out_vec(total_elements);

    if (total_elements > 0) {
        if (!storage || !addend.storage) {
            throw std::runtime_error("NDArray storage is unallocated");
        }
        const T* a_ptr = storage->data.data() + offset;
        const T* b_ptr = addend.storage->data.data() + addend.offset;

        // 5. Accumulate element values from respective storage offsets
        std::vector<size_t> coord(max_ndim, 0);
        for (size_t out_idx = 0; out_idx < total_elements; ++out_idx) {
            size_t off_a = 0;
            size_t off_b = 0;
            for (size_t d = 0; d < max_ndim; ++d) {
                off_a += coord[d] * eff_strides_a[d];
                off_b += coord[d] * eff_strides_b[d];
            }
            out_vec[out_idx] = a_ptr[off_a] + b_ptr[off_b];

            if (max_ndim > 0) {
                for (int d = static_cast<int>(max_ndim) - 1; d >= 0; --d) {
                    coord[d]++;
                    if (coord[d] < out_shape[d]) {
                        break;
                    }
                    coord[d] = 0;
                }
            }
        }
    }

    return ndarray<T>(out_vec, out_shape);
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
