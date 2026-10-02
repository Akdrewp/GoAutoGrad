#include "ndarray.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

/**
 * @brief Computes standard row-major (C-contiguous) strides for a given shape.
 * 
 * In a standard C-contiguous layout, the innermost dimension has a stride of 1,
 * and each preceding dimension's stride is the product of all following dimension sizes:
 *   stride[i] = stride[i + 1] * shape[i + 1]
 * 
 * 1. Checks if the shape is empty (0-D scalar) and returns empty strides.
 * 2. Initializes the stride vector with 1s matching the rank of the shape.
 * 3. Iterates backwards from the second-to-innermost dimension, multiplying the
 *    subsequent stride by the subsequent dimension size.
 * 4. Returns the computed row-major strides.
 * 
 * @param shape Shape dimensions vector.
 * @return Row-major strides vector for each dimension.
 */
template <typename T>
std::vector<size_t> ndarray<T>::default_strides(const std::vector<size_t>& shape) {
    // 1. Check if shape is empty (scalar)
    if (shape.empty()) {
        return {};
    }
    // 2. Initialize stride vector with 1s
    std::vector<size_t> str(shape.size(), 1);
    // 3. Iteratively compute strides moving backwards
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        str[i] = str[i + 1] * shape[i + 1];
    }
    // 4. Return computed strides
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

/**
 * @brief Applies an element-wise unary transformation across an ndarray.
 * @param in Input ndarray.
 * @param op Unary callable taking an element and returning transformed value.
 * @return New ndarray with transformed elements and standard strides.
 * @throws std::runtime_error If input storage is unallocated.
 */
template <typename T, typename UnaryOp>
inline ndarray<T> apply_unary_op(const ndarray<T>& in, UnaryOp op) {
    if (in.isEmpty()) {
        return ndarray<T>(std::vector<T>{}, in.shape);
    }
    if (!in.storage) {
        throw std::runtime_error("NDArray storage is unallocated");
    }

    size_t total_elements = in.size();
    std::vector<T> out_vec(total_elements);

    if (in.is_contiguous()) {
        const T* src = in.storage->data.data() + in.offset;
        for (size_t i = 0; i < total_elements; ++i) {
            out_vec[i] = op(src[i]);
        }
    } else {
        size_t rank = in.shape.size();
        std::vector<size_t> coord(rank, 0);
        const T* src = in.storage->data.data() + in.offset;
        for (size_t i = 0; i < total_elements; ++i) {
            size_t off = 0;
            for (size_t d = 0; d < rank; ++d) {
                off += coord[d] * in.strides[d];
            }
            out_vec[i] = op(src[off]);
            advance_coordinate(coord, in.shape);
        }
    }

    return ndarray<T>(out_vec, in.shape);
}

/**
 * @brief Applies an element-wise backward transformation across input and upstream gradient.
 * @param in Forward input ndarray.
 * @param grad_output Upstream gradient ndarray.
 * @param op Backward callable taking (input_element, grad_element).
 * @return New ndarray with computed gradient elements and standard strides.
 * @throws std::invalid_argument If grad_output shape does not match input shape.
 * @throws std::runtime_error If either storage buffer is unallocated.
 */
template <typename T, typename BackwardOp>
inline ndarray<T> apply_backward_op(
    const ndarray<T>& in,
    const ndarray<T>& grad_output,
    BackwardOp op
) {
    if (in.shape != grad_output.shape) {
        throw std::invalid_argument(
            "shape mismatch: grad_output shape does not match array shape"
        );
    }
    if (in.isEmpty()) {
        return ndarray<T>(std::vector<T>{}, in.shape);
    }
    if (!in.storage || !grad_output.storage) {
        throw std::runtime_error("NDArray storage is unallocated");
    }

    size_t total_elements = in.size();
    std::vector<T> out_vec(total_elements);

    if (in.is_contiguous() && grad_output.is_contiguous()) {
        const T* src = in.storage->data.data() + in.offset;
        const T* grad = grad_output.storage->data.data() + grad_output.offset;
        for (size_t i = 0; i < total_elements; ++i) {
            out_vec[i] = op(src[i], grad[i]);
        }
    } else {
        size_t rank = in.shape.size();
        std::vector<size_t> coord(rank, 0);
        const T* src = in.storage->data.data() + in.offset;
        const T* grad = grad_output.storage->data.data() + grad_output.offset;
        for (size_t i = 0; i < total_elements; ++i) {
            size_t off_src = 0;
            size_t off_grad = 0;
            for (size_t d = 0; d < rank; ++d) {
                off_src += coord[d] * in.strides[d];
                off_grad += coord[d] * grad_output.strides[d];
            }
            out_vec[i] = op(src[off_src], grad[off_grad]);
            advance_coordinate(coord, in.shape);
        }
    }

    return ndarray<T>(out_vec, in.shape);
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

/**
 * @brief Applies element-wise rectified linear unit (ReLU): max(0, x).
 * @return New ndarray with ReLU applied to each element.
 */
template <typename T>
ndarray<T> ndarray<T>::relu() const {
    return apply_unary_op(*this, [](T x) {
        return x > static_cast<T>(0) ? x : static_cast<T>(0);
    });
}

/**
 * @brief Computes gradient for ReLU activation element-wise: grad_out * (x > 0 ? 1 : 0).
 * @param grad_output Incoming upstream gradient array.
 * @return New ndarray containing the computed gradients.
 * @throws std::invalid_argument If grad_output shape does not match array shape.
 */
template <typename T>
ndarray<T> ndarray<T>::relu_backward(const ndarray<T>& grad_output) const {
    return apply_backward_op(*this, grad_output, [](T x, T grad) {
        return x > static_cast<T>(0) ? grad : static_cast<T>(0);
    });
}

/**
 * @brief Applies element-wise hyperbolic tangent (tanh).
 * @return New ndarray with tanh applied to each element.
 */
template <typename T>
ndarray<T> ndarray<T>::tanh() const {
    return apply_unary_op(*this, [](T x) {
        return std::tanh(x);
    });
}

/**
 * @brief Computes gradient for hyperbolic tangent (tanh) element-wise: grad_out * (1 - tanh(x)^2).
 * @param grad_output Incoming upstream gradient array.
 * @return New ndarray containing the computed gradients.
 * @throws std::invalid_argument If grad_output shape does not match array shape.
 */
template <typename T>
ndarray<T> ndarray<T>::tanh_backward(const ndarray<T>& grad_output) const {
    return apply_backward_op(*this, grad_output, [](T x, T grad) {
        T t = std::tanh(x);
        return grad * (static_cast<T>(1) - t * t);
    });
}

/**
 * @brief Applies element-wise sigmoid activation: 1 / (1 + exp(-x)).
 * @return New ndarray with sigmoid applied to each element.
 */
template <typename T>
ndarray<T> ndarray<T>::sigmoid() const {
    return apply_unary_op(*this, [](T x) {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    });
}

/**
 * @brief Computes gradient for sigmoid element-wise: grad_out * sigmoid(x) * (1 - sigmoid(x)).
 * @param grad_output Incoming upstream gradient array.
 * @return New ndarray containing the computed gradients.
 * @throws std::invalid_argument If grad_output shape does not match array shape.
 */
template <typename T>
ndarray<T> ndarray<T>::sigmoid_backward(const ndarray<T>& grad_output) const {
    return apply_backward_op(*this, grad_output, [](T x, T grad) {
        T s = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
        return grad * s * (static_cast<T>(1) - s);
    });
}

/**
 * @brief Applies element-wise leaky rectified linear unit (LeakyReLU): x if x > 0 else alpha * x.
 * @param alpha Slope for negative inputs. Defaults to 0.01.
 * @return New ndarray with LeakyReLU applied to each element.
 */
template <typename T>
ndarray<T> ndarray<T>::leaky_relu(T alpha) const {
    return apply_unary_op(*this, [alpha](T x) {
        return x > static_cast<T>(0) ? x : alpha * x;
    });
}

/**
 * @brief Computes gradient for LeakyReLU element-wise: grad_out * (x > 0 ? 1 : alpha).
 * @param grad_output Incoming upstream gradient array.
 * @param alpha Slope for negative inputs. Defaults to 0.01.
 * @return New ndarray containing the computed gradients.
 * @throws std::invalid_argument If grad_output shape does not match array shape.
 */
template <typename T>
ndarray<T> ndarray<T>::leaky_relu_backward(const ndarray<T>& grad_output, T alpha) const {
    return apply_backward_op(*this, grad_output, [alpha](T x, T grad) {
        return x > static_cast<T>(0) ? grad : grad * alpha;
    });
}

/**
 * @brief Computes the sum of elements over a specified dimension.
 * @param dim Dimension along which to sum. Negative values index from the end.
 * @param keepdim Whether the output array retains the reduced dimension as size 1.
 * @return New ndarray with summed values.
 * @throws std::out_of_range If dim is out of range [-ndim, ndim-1].
 */
template <typename T>
ndarray<T> ndarray<T>::sum(int dim, bool keepdim) const {
    size_t rank = shape.size();
    if (rank == 0) {
        if (dim == 0 || dim == -1) {
            return *this;
        }
        throw std::out_of_range("Dimension out of range for 0-D array: " + std::to_string(dim));
    }

    int norm_dim = dim;
    if (norm_dim < 0) {
        norm_dim += static_cast<int>(rank);
    }
    if (norm_dim < 0 || norm_dim >= static_cast<int>(rank)) {
        throw std::out_of_range("Dimension out of range: " + std::to_string(dim));
    }
    size_t axis = static_cast<size_t>(norm_dim);

    std::vector<size_t> out_shape;
    if (keepdim) {
        out_shape = shape;
        out_shape[axis] = 1;
    } else {
        out_shape.reserve(rank > 1 ? rank - 1 : 0);
        for (size_t i = 0; i < rank; ++i) {
            if (i != axis) {
                out_shape.push_back(shape[i]);
            }
        }
    }

    size_t out_num_el = num_elements(out_shape);
    std::vector<T> out_data(out_num_el, static_cast<T>(0));
    std::vector<size_t> out_strides = default_strides(out_shape);

    if (size() == 0 || !storage) {
        return ndarray<T>(out_data, out_shape);
    }

    std::vector<size_t> in_coord(rank, 0);
    std::vector<size_t> out_coord(out_shape.size(), 0);
    size_t total_in = size();

    for (size_t step = 0; step < total_in; ++step) {
        if (keepdim) {
            for (size_t i = 0; i < rank; ++i) {
                out_coord[i] = (i == axis) ? 0 : in_coord[i];
            }
        } else {
            size_t c_idx = 0;
            for (size_t i = 0; i < rank; ++i) {
                if (i != axis) {
                    out_coord[c_idx++] = in_coord[i];
                }
            }
        }

        size_t out_offset = 0;
        for (size_t i = 0; i < out_coord.size(); ++i) {
            out_offset += out_coord[i] * out_strides[i];
        }

        size_t in_offset = index_to_offset(in_coord);
        out_data[out_offset] += storage->data[in_offset];
        advance_coordinate(in_coord, shape);
    }

    return ndarray<T>(out_data, out_shape);
}

/**
 * @brief Computes the total sum of all elements in the array.
 * @return New ndarray containing the scalar total sum.
 */
template <typename T>
ndarray<T> ndarray<T>::sum() const {
    if (size() == 0 || !storage) {
        return ndarray<T>(std::vector<T>{static_cast<T>(0)}, {});
    }

    T total = static_cast<T>(0);
    if (is_contiguous()) {
        for (size_t i = 0; i < size(); ++i) {
            total += storage->data[offset + i];
        }
    } else {
        std::vector<size_t> in_coord(shape.size(), 0);
        size_t total_in = size();
        for (size_t step = 0; step < total_in; ++step) {
            total += storage->data[index_to_offset(in_coord)];
            advance_coordinate(in_coord, shape);
        }
    }

    return ndarray<T>(std::vector<T>{total}, {});
}

// Explicit template instantiations
template struct Storage<float>;
template struct Storage<double>;
template struct Storage<int>;

template class ndarray<float>;
template class ndarray<double>;
template class ndarray<int>;
