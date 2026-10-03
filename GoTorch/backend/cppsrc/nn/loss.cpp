#include "nn/loss.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gotorch/kernels.hpp"
#include "gotorch/ndarray.hpp"
#include "gotorch/shape_utils.hpp"

namespace gotorch {
namespace nn {

Reduction parse_reduction(const std::string& reduction_str) {
    if (reduction_str == "mean") {
        return Reduction::Mean;
    }
    if (reduction_str == "sum") {
        return Reduction::Sum;
    }
    if (reduction_str == "none") {
        return Reduction::None;
    }
    throw std::invalid_argument(
        "Invalid reduction mode: '" + reduction_str + "'. Expected 'mean', 'sum', or 'none'."
    );
}

std::string reduction_to_string(Reduction reduction) {
    switch (reduction) {
        case Reduction::Mean:
            return "mean";
        case Reduction::Sum:
            return "sum";
        case Reduction::None:
            return "none";
    }
    return "unknown";
}

template <typename T>
ndarray<T> mse_loss(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    Reduction reduction
) {
    if (prediction.isEmpty() || target.isEmpty()) {
        if (reduction == Reduction::None) {
            return ndarray<T>();
        }
        return ndarray<T>(std::vector<T>{static_cast<T>(0)}, {});
    }

    ndarray<T> sq_diff = kernels::for_each_binary(
        prediction, target, [](T p, T t) {
            T diff = p - t;
            return diff * diff;
        }
    );

    if (reduction == Reduction::None) {
        return sq_diff;
    }

    T total = kernels::reduce_all(sq_diff, std::plus<T>{}, static_cast<T>(0));
    if (reduction == Reduction::Mean) {
        T mean = sq_diff.size() > 0 ? (total / static_cast<T>(sq_diff.size())) : static_cast<T>(0);
        return ndarray<T>(std::vector<T>{mean}, {});
    }
    return ndarray<T>(std::vector<T>{total}, {});
}

template <typename T>
ndarray<T> mse_loss(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const std::string& reduction
) {
    return mse_loss(prediction, target, parse_reduction(reduction));
}

template <typename T>
ndarray<T> mse_loss_backward(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const ndarray<T>& grad_output,
    Reduction reduction
) {
    // Determine broadcasted total elements N for mean scaling
    size_t ndim_p = prediction.shape.size();
    size_t ndim_t = target.shape.size();
    size_t max_ndim = std::max(ndim_p, ndim_t);
    std::vector<size_t> padded_p = shape_utils::pad_shape(prediction.shape, max_ndim);
    std::vector<size_t> padded_t = shape_utils::pad_shape(target.shape, max_ndim);
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(padded_p, padded_t);
    size_t total_elements = shape_utils::num_elements(out_shape);

    T scale = static_cast<T>(2);
    if (reduction == Reduction::Mean) {
        scale = total_elements > 0 ? (static_cast<T>(2) / static_cast<T>(total_elements)) : static_cast<T>(0);
    }

    bool grad_is_scalar = grad_output.isEmpty() || grad_output.size() == 1;
    ndarray<T> full_grad;

    if (grad_is_scalar) {
        T scalar_grad = !grad_output.isEmpty() ? grad_output[0] : static_cast<T>(1);
        T total_scale = scale * scalar_grad;
        full_grad = kernels::for_each_binary(
            prediction, target, [total_scale](T p, T t) {
                return total_scale * (p - t);
            }
        );
    } else {
        ndarray<T> diff = kernels::for_each_binary(
            prediction, target, [scale](T p, T t) {
                return scale * (p - t);
            }
        );
        full_grad = kernels::for_each_binary(
            diff, grad_output, [](T d, T g) {
                return d * g;
            }
        );
    }

    return shape_utils::unbroadcast(full_grad, prediction.shape);
}

template <typename T>
ndarray<T> mse_loss_backward(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const ndarray<T>& grad_output,
    const std::string& reduction
) {
    return mse_loss_backward(prediction, target, grad_output, parse_reduction(reduction));
}

// Explicit template instantiations
template ndarray<float> mse_loss<float>(const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> mse_loss<float>(const ndarray<float>&, const ndarray<float>&, const std::string&);
template ndarray<float> mse_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> mse_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, const std::string&);
template class MSELoss<float>;

template ndarray<double> mse_loss<double>(const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> mse_loss<double>(const ndarray<double>&, const ndarray<double>&, const std::string&);
template ndarray<double> mse_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> mse_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, const std::string&);
template class MSELoss<double>;

}  // namespace nn
}  // namespace gotorch
