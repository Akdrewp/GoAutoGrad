#include "gotorch/shape_utils.hpp"
#include "gotorch/ndarray.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace gotorch {
namespace shape_utils {

size_t num_elements(const std::vector<size_t>& shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}

std::vector<size_t> default_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    std::vector<size_t> str(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        str[i] = str[i + 1] * shape[i + 1];
    }
    return str;
}

size_t index_to_offset(
    const std::vector<size_t>& indices,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    size_t base_offset
) {
    if (indices.size() != shape.size()) {
        throw std::out_of_range(
            "Index dimensional mismatch: expected " +
            std::to_string(shape.size()) + " dims, got " +
            std::to_string(indices.size())
        );
    }
    size_t off = base_offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        off += indices[i] * strides[i];
    }
    return off;
}

size_t compute_coordinate_offset(
    const std::vector<size_t>& coord,
    const std::vector<size_t>& strides
) {
    size_t off = 0;
    for (size_t i = 0; i < coord.size(); ++i) {
        off += coord[i] * strides[i];
    }
    return off;
}

void advance_coordinate(
    std::vector<size_t>& coord,
    const std::vector<size_t>& shape
) {
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        coord[d]++;
        if (coord[d] < shape[d]) {
            break;
        }
        coord[d] = 0;
    }
}

std::vector<size_t> pad_shape(
    const std::vector<size_t>& shape,
    size_t target_rank
) {
    std::vector<size_t> padded(target_rank, 1);
    size_t pad = target_rank - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        padded[pad + i] = shape[i];
    }
    return padded;
}

std::vector<size_t> broadcast_shapes(
    const std::vector<size_t>& padded_a,
    const std::vector<size_t>& padded_b
) {
    size_t max_ndim = padded_a.size();
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
    return out_shape;
}

std::vector<size_t> compute_broadcast_strides(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    size_t target_rank
) {
    std::vector<size_t> eff_strides(target_rank, 0);
    size_t pad = target_rank - shape.size();

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

size_t normalize_axis(int dim, size_t rank) {
    if (rank == 0) {
        if (dim == 0 || dim == -1) {
            return 0;
        }
        throw std::out_of_range("Dimension out of range for 0-D array: " + std::to_string(dim));
    }
    int norm = dim;
    if (norm < 0) {
        norm += static_cast<int>(rank);
    }
    if (norm < 0 || norm >= static_cast<int>(rank)) {
        throw std::out_of_range("Dimension out of range: " + std::to_string(dim));
    }
    return static_cast<size_t>(norm);
}

std::vector<size_t> compute_reduced_shape(
    const std::vector<size_t>& shape,
    size_t axis,
    bool keepdim
) {
    if (keepdim) {
        std::vector<size_t> out_shape = shape;
        out_shape[axis] = 1;
        return out_shape;
    }
    std::vector<size_t> out_shape;
    out_shape.reserve(shape.size() > 1 ? shape.size() - 1 : 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != axis) {
            out_shape.push_back(shape[i]);
        }
    }
    return out_shape;
}

void map_reduced_coordinate(
    const std::vector<size_t>& in_coord,
    size_t axis,
    bool keepdim,
    std::vector<size_t>& out_coord
) {
    if (keepdim) {
        for (size_t i = 0; i < in_coord.size(); ++i) {
            out_coord[i] = (i == axis) ? 0 : in_coord[i];
        }
    } else {
        size_t idx = 0;
        for (size_t i = 0; i < in_coord.size(); ++i) {
            if (i != axis) {
                out_coord[idx++] = in_coord[i];
            }
        }
    }
}

template <typename T>
ndarray<T> unbroadcast(const ndarray<T>& in, const std::vector<size_t>& target_shape) {
    if (in.isEmpty() || in.shape == target_shape) {
        return in;
    }
    if (target_shape.empty()) {
        return in.sum();
    }

    ndarray<T> curr = in;

    // 1. Sum over prepended leading dimensions
    if (curr.shape.size() > target_shape.size()) {
        size_t num_leading = curr.shape.size() - target_shape.size();
        for (size_t i = 0; i < num_leading; ++i) {
            curr = curr.sum(0, false);
        }
    }

    // 2. Sum over dimensions where target dimension is 1 but current dimension > 1
    for (size_t dim_idx = 0; dim_idx < target_shape.size(); ++dim_idx) {
        if (target_shape[dim_idx] == 1 && curr.shape[dim_idx] > 1) {
            curr = curr.sum(static_cast<int>(dim_idx), true);
        }
    }

    return curr;
}

template ndarray<float> unbroadcast<float>(const ndarray<float>&, const std::vector<size_t>&);
template ndarray<double> unbroadcast<double>(const ndarray<double>&, const std::vector<size_t>&);
template ndarray<int> unbroadcast<int>(const ndarray<int>&, const std::vector<size_t>&);

}  // namespace shape_utils
}  // namespace gotorch
