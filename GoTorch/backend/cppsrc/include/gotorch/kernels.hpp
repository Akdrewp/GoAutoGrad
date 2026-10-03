#ifndef GOTORCH_KERNELS_HPP
#define GOTORCH_KERNELS_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gotorch/ndarray.hpp"
#include "gotorch/shape_utils.hpp"

namespace gotorch {
namespace kernels {

// ============================================================================
// Execution Iterators
// ============================================================================

/**
 * @brief Applies an element-wise unary transformation across an ndarray.
 *
 * @param in Input ndarray.
 * @param op Unary callable taking an element value and returning the transformed value.
 * @return New ndarray with transformed elements and standard contiguous strides.
 */
template <typename T, typename UnaryOp>
ndarray<T> for_each_unary(const ndarray<T>& in, UnaryOp op) {
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
            size_t off = shape_utils::compute_coordinate_offset(coord, in.strides);
            out_vec[i] = op(src[off]);
            shape_utils::advance_coordinate(coord, in.shape);
        }
    }

    return ndarray<T>(out_vec, in.shape);
}

/**
 * @brief Computes an element-wise binary operation across two arrays with broadcasting.
 *
 * @param a First operand.
 * @param b Second operand.
 * @param op Binary callable taking (elem_a, elem_b) and returning output value.
 * @return New ndarray with broadcasted output shape.
 */
template <typename T, typename BinaryOp>
ndarray<T> for_each_binary(const ndarray<T>& a, const ndarray<T>& b, BinaryOp op) {
    if (a.isEmpty() || b.isEmpty()) {
        return ndarray<T>();
    }
    if (!a.storage || !b.storage) {
        throw std::runtime_error("NDArray storage is unallocated");
    }

    size_t ndim_a = a.shape.size();
    size_t ndim_b = b.shape.size();
    size_t max_ndim = std::max(ndim_a, ndim_b);

    std::vector<size_t> padded_a = shape_utils::pad_shape(a.shape, max_ndim);
    std::vector<size_t> padded_b = shape_utils::pad_shape(b.shape, max_ndim);
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(padded_a, padded_b);

    std::vector<size_t> eff_strides_a = shape_utils::compute_broadcast_strides(a.shape, a.strides, max_ndim);
    std::vector<size_t> eff_strides_b = shape_utils::compute_broadcast_strides(b.shape, b.strides, max_ndim);

    size_t total_elements = shape_utils::num_elements(out_shape);
    std::vector<T> out_vec(total_elements);

    const T* data_a = a.storage->data.data() + a.offset;
    const T* data_b = b.storage->data.data() + b.offset;

    std::vector<size_t> coord(max_ndim, 0);
    for (size_t i = 0; i < total_elements; ++i) {
        size_t off_a = shape_utils::compute_coordinate_offset(coord, eff_strides_a);
        size_t off_b = shape_utils::compute_coordinate_offset(coord, eff_strides_b);
        out_vec[i] = op(data_a[off_a], data_b[off_b]);
        shape_utils::advance_coordinate(coord, out_shape);
    }

    return ndarray<T>(out_vec, out_shape);
}

/**
 * @brief Computes backward gradient propagation across input and upstream gradient arrays.
 *
 * @param in Forward input array.
 * @param grad_output Upstream gradient array.
 * @param op Backward callable taking (input_element, grad_element).
 * @return New ndarray with computed gradient values.
 */
template <typename T, typename BackwardOp>
ndarray<T> for_each_backward(
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
            size_t off_src = shape_utils::compute_coordinate_offset(coord, in.strides);
            size_t off_grad = shape_utils::compute_coordinate_offset(coord, grad_output.strides);
            out_vec[i] = op(src[off_src], grad[off_grad]);
            shape_utils::advance_coordinate(coord, in.shape);
        }
    }

    return ndarray<T>(out_vec, in.shape);
}

/**
 * @brief Reduces an ndarray along a specified axis using an associative binary operator.
 *
 * @param in Input ndarray.
 * @param dim Axis to reduce.
 * @param keepdim Whether to retain reduced dimension with size 1.
 * @param op Reduction associative callable taking (acc, val).
 * @param init_val Initial value for reduction.
 * @return Reduced ndarray.
 */
template <typename T, typename ReduceOp>
ndarray<T> reduce_along_axis(
    const ndarray<T>& in,
    int dim,
    bool keepdim,
    ReduceOp op,
    T init_val
) {
    size_t rank = in.shape.size();
    if (rank == 0) {
        if (dim == 0 || dim == -1) {
            return in;
        }
        throw std::out_of_range("Dimension out of range for 0-D array: " + std::to_string(dim));
    }

    size_t axis = shape_utils::normalize_axis(dim, rank);
    std::vector<size_t> out_shape = shape_utils::compute_reduced_shape(in.shape, axis, keepdim);
    size_t out_num_el = shape_utils::num_elements(out_shape);
    std::vector<T> out_data(out_num_el, init_val);
    std::vector<size_t> out_strides = shape_utils::default_strides(out_shape);

    if (in.size() == 0 || !in.storage) {
        return ndarray<T>(out_data, out_shape);
    }

    std::vector<size_t> in_coord(rank, 0);
    std::vector<size_t> out_coord(out_shape.size(), 0);
    size_t total_in = in.size();

    for (size_t step = 0; step < total_in; ++step) {
        shape_utils::map_reduced_coordinate(in_coord, axis, keepdim, out_coord);
        size_t out_offset = shape_utils::compute_coordinate_offset(out_coord, out_strides);
        size_t in_offset = in.index_to_offset(in_coord);
        out_data[out_offset] = op(out_data[out_offset], in.storage->data[in_offset]);
        shape_utils::advance_coordinate(in_coord, in.shape);
    }

    return ndarray<T>(out_data, out_shape);
}

/**
 * @brief Reduces all elements of an ndarray into a scalar using an associative binary operator.
 *
 * @param in Input ndarray.
 * @param op Reduction associative callable taking (acc, val).
 * @param init_val Initial value for reduction.
 * @return Scalar reduced value.
 */
template <typename T, typename ReduceOp>
T reduce_all(const ndarray<T>& in, ReduceOp op, T init_val) {
    if (in.size() == 0 || !in.storage) {
        return init_val;
    }

    T total = init_val;
    if (in.is_contiguous()) {
        for (size_t i = 0; i < in.size(); ++i) {
            total = op(total, in.storage->data[in.offset + i]);
        }
    } else {
        std::vector<size_t> in_coord(in.shape.size(), 0);
        size_t total_in = in.size();
        for (size_t step = 0; step < total_in; ++step) {
            total = op(total, in.storage->data[in.index_to_offset(in_coord)]);
            shape_utils::advance_coordinate(in_coord, in.shape);
        }
    }
    return total;
}

// ============================================================================
// Mathematical Operations & Functions
// ============================================================================

template <typename T>
ndarray<T> add(const ndarray<T>& a, const ndarray<T>& b) {
    return for_each_binary(a, b, std::plus<T>{});
}

template <typename T>
ndarray<T> sub(const ndarray<T>& a, const ndarray<T>& b) {
    return for_each_binary(a, b, std::minus<T>{});
}

template <typename T>
ndarray<T> mul(const ndarray<T>& a, const ndarray<T>& b);

template <typename T>
ndarray<T> matmul(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument(
            "matmul requires 2D arrays, got " +
            std::to_string(a.shape.size()) + "D and " +
            std::to_string(b.shape.size()) + "D"
        );
    }

    size_t M = a.shape[0];
    size_t K1 = a.shape[1];
    size_t K2 = b.shape[0];
    size_t N = b.shape[1];

    if (K1 != K2) {
        throw std::invalid_argument(
            "shape mismatch: inner dimensions (" +
            std::to_string(K1) + " and " + std::to_string(K2) + ") must match"
        );
    }

    std::vector<T> out_vec(M * N, static_cast<T>(0));

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T dot_sum = static_cast<T>(0);
            for (size_t k = 0; k < K1; ++k) {
                dot_sum += a({i, k}) * b({k, j});
            }
            out_vec[(i * N) + j] = dot_sum;
        }
    }

    return ndarray<T>(out_vec, {M, N});
}

template <typename T>
ndarray<T> relu(const ndarray<T>& a) {
    return for_each_unary(a, [](T x) {
        return x > static_cast<T>(0) ? x : static_cast<T>(0);
    });
}

template <typename T>
ndarray<T> relu_backward(const ndarray<T>& a, const ndarray<T>& grad_output) {
    return for_each_backward(a, grad_output, [](T x, T grad) {
        return x > static_cast<T>(0) ? grad : static_cast<T>(0);
    });
}

template <typename T>
ndarray<T> tanh(const ndarray<T>& a) {
    return for_each_unary(a, [](T x) {
        return std::tanh(x);
    });
}

template <typename T>
ndarray<T> tanh_backward(const ndarray<T>& a, const ndarray<T>& grad_output) {
    return for_each_backward(a, grad_output, [](T x, T grad) {
        T th = std::tanh(x);
        return grad * (static_cast<T>(1) - (th * th));
    });
}

template <typename T>
ndarray<T> sigmoid(const ndarray<T>& a) {
    return for_each_unary(a, [](T x) {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    });
}

template <typename T>
ndarray<T> sigmoid_backward(const ndarray<T>& a, const ndarray<T>& grad_output) {
    return for_each_backward(a, grad_output, [](T x, T grad) {
        T s = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
        return grad * s * (static_cast<T>(1) - s);
    });
}

template <typename T>
ndarray<T> leaky_relu(const ndarray<T>& a, T alpha = static_cast<T>(0.01)) {
    return for_each_unary(a, [alpha](T x) {
        return x > static_cast<T>(0) ? x : alpha * x;
    });
}

template <typename T>
ndarray<T> leaky_relu_backward(
    const ndarray<T>& a,
    const ndarray<T>& grad_output,
    T alpha = static_cast<T>(0.01)
) {
    return for_each_backward(a, grad_output, [alpha](T x, T grad) {
        return x > static_cast<T>(0) ? grad : grad * alpha;
    });
}

template <typename T>
ndarray<T> sum(const ndarray<T>& a, int dim, bool keepdim = false) {
    return reduce_along_axis(a, dim, keepdim, std::plus<T>{}, static_cast<T>(0));
}

template <typename T>
ndarray<T> sum(const ndarray<T>& a) {
    T total = reduce_all(a, std::plus<T>{}, static_cast<T>(0));
    return ndarray<T>(std::vector<T>{total}, {});
}

}  // namespace kernels
}  // namespace gotorch

#endif  // GOTORCH_KERNELS_HPP
