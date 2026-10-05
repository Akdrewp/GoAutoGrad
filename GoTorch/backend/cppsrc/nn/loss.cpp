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

/**
 * @brief Parses reduction mode from string representation.
 *
 * @param reduction_str String name of reduction ("mean", "sum", "none").
 * @return Parsed Reduction enum value.
 * @throw std::invalid_argument If reduction string is unrecognized.
 */
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

/**
 * @brief Converts Reduction enum to its string representation.
 *
 * @param reduction Reduction enum value.
 * @return String representation ("mean", "sum", "none").
 */
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

/**
 * @brief Applies reduction operation to an unreduced loss tensor.
 *
 * 1. Return unreduced loss tensor directly if reduction is 'none'.
 * 2. Return zero scalar if input tensor contains no elements.
 * 3. Reduce across all elements and divide by total elements if reduction is 'mean'.
 *
 * @param loss Unreduced loss ndarray.
 * @param reduction Reduction mode (Mean, Sum, None).
 * @return Reduced loss ndarray (scalar for Mean/Sum, matching shape for None).
 */
template <typename T>
ndarray<T> apply_reduction(const ndarray<T>& loss, Reduction reduction) {
    // Step 1: Return unreduced loss tensor directly if reduction is 'none'
    if (reduction == Reduction::None) {
        return loss;
    }
    // Step 2: Return zero scalar if input tensor contains no elements
    if (loss.isEmpty()) {
        return ndarray<T>(std::vector<T>{static_cast<T>(0)}, {});
    }
    // Step 3: Reduce across all elements and divide by size if reduction is 'mean'
    T total = kernels::reduce_all(loss, std::plus<T>{}, static_cast<T>(0));
    if (reduction == Reduction::Mean) {
        return ndarray<T>(std::vector<T>{total / static_cast<T>(loss.size())}, {});
    }
    return ndarray<T>(std::vector<T>{total}, {});
}

/**
 * @brief Computes Mean Squared Error (MSE) loss between prediction and target.
 *
 * Evaluates element-wise squared differences (p - t)^2 and reduces according to reduction mode.
 *
 * 1. Compute element-wise squared difference (p - t)^2 via binary kernel.
 * 2. Apply requested reduction to squared differences.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values with broadcast-compatible shape.
 * @param reduction Reduction mode (Mean, Sum, None).
 * @return Computed loss ndarray (scalar for Mean/Sum, matching shape for None).
 * @throw std::invalid_argument If shapes cannot be broadcast together.
 */
template <typename T>
ndarray<T> mse_loss(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    Reduction reduction
) {
    // Step 1: Compute element-wise squared difference (p - t)^2
    ndarray<T> sq_diff = kernels::for_each_binary(
        prediction, target, [](T p, T t) {
            T diff = p - t;
            return diff * diff;
        }
    );
    // Step 2: Apply requested reduction
    return apply_reduction(sq_diff, reduction);
}

/**
 * @brief Computes MSE loss accepting string reduction mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Computed loss ndarray.
 * @throw std::invalid_argument If reduction string is unrecognized or shapes cannot broadcast.
 */
template <typename T>
ndarray<T> mse_loss(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const std::string& reduction
) {
    return mse_loss(prediction, target, parse_reduction(reduction));
}

/**
 * @brief Computes backward gradient of MSE loss with respect to prediction.
 *
 * Evaluates the analytical gradient:
 * - None: 2 * (p - t) * grad_output
 * - Sum:  2 * (p - t) * grad_output
 * - Mean: (2 / N) * (p - t) * grad_output
 * and automatically unbroadcasts to match prediction operand shape.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction Reduction mode applied in forward pass.
 * @return Gradient ndarray with respect to prediction matching prediction.shape.
 * @throw std::invalid_argument If shapes cannot broadcast together.
 */
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
    if (grad_is_scalar) {
        T scalar_grad = !grad_output.isEmpty() ? grad_output[0] : static_cast<T>(1);
        T total_scale = scale * scalar_grad;
        ndarray<T> full_grad = kernels::for_each_binary(
            prediction, target, [total_scale](T p, T t) {
                return total_scale * (p - t);
            }
        );
        return shape_utils::unbroadcast(full_grad, prediction.shape);
    }

    ndarray<T> diff = kernels::for_each_binary(
        prediction, target, [scale](T p, T t) {
            return scale * (p - t);
        }
    );
    ndarray<T> full_grad = kernels::for_each_binary(
        diff, grad_output, [](T d, T g) {
            return d * g;
        }
    );
    return shape_utils::unbroadcast(full_grad, prediction.shape);
}

/**
 * @brief Computes backward gradient of MSE loss accepting string reduction mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Gradient ndarray with respect to prediction.
 * @throw std::invalid_argument If reduction is unrecognized or shapes cannot broadcast.
 */
template <typename T>
ndarray<T> mse_loss_backward(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const ndarray<T>& grad_output,
    const std::string& reduction
) {
    return mse_loss_backward(prediction, target, grad_output, parse_reduction(reduction));
}


/**
 * @brief Computes Binary Cross Entropy (BCE) loss between prediction and target.
 *
 * The loss for one output is:
 * There is some binary class y = {0, 1}
 * and some prediction p = P(y = 1) so P(y = 0) = 1 - p.
 *
 * Then an objective function that measures a models prediction = y_p
 * y_p^y * (1 - y_p)^(1-y).
 * Which just measures binomial approximation for the prediction.
 * 
 * A loss function can be made by taking the log
 * ln(l) = y * ln(y_p) + (1-y) * ln(1-y_p)
 * l = - ( y * ln(y_p) + (1-y) * ln(1-y_p) )
 * Here, where y != 0, the prediction y_p should close to 0
 * Else y == 1: ln(y_p), where y_p < 0, should be close to 1
 *
 * The loss over the batch is 
 * sum for y in batch: 
 *  l = - ( y * ln(y_p) + (1-y) * ln(1-y_p) )
 *
 * 1. Compute element-wise loss via standard equation: -(y * ln(p) + (1 - y) * ln(1 - p)).
 * 2. Apply requested reduction to element-wise losses.
 *
 * @param in_features Input predicted probabilities.
 * @param true_features Ground truth binary targets (0 or 1).
 * @param reduction Reduction mode (Mean, Sum, None).
 * @return Computed loss ndarray (scalar for Mean/Sum, matching shape for None).
 * @throw std::invalid_argument If shapes cannot broadcast together.
 */
template <typename T>
ndarray<T> BCELoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    Reduction reduction
) {
    // Step 1: Compute element-wise loss via standard equation
    ndarray<T> loss_elements = kernels::for_each_binary(
        in_features, true_features, [](T p, T y) {
            return -((y * std::log(p)) + ((static_cast<T>(1) - y) * std::log(static_cast<T>(1) - p)));
        }
    );
    // Step 2: Apply requested reduction
    return apply_reduction(loss_elements, reduction);
}

/**
 * @brief Computes BCE loss accepting string reduction mode.
 *
 * @param in_features Input predicted probabilities.
 * @param true_features Ground truth binary targets.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Computed loss ndarray.
 * @throw std::invalid_argument If reduction string is unrecognized or shapes cannot broadcast.
 */
template <typename T>
ndarray<T> BCELoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const std::string& reduction
) {
    return BCELoss(in_features, true_features, parse_reduction(reduction));
}

/**
 * @brief Computes backward gradient of BCE loss with respect to in_features.
 *
 * 1. Compute broadcast shape and reduction scale factor.
 * 2. Evaluate analytical gradient: (p - y) / (p * (1 - p)) * scale * grad_output.
 * 3. Unbroadcast gradient back to in_features operand shape.
 *
 * @param in_features Input predicted probabilities.
 * @param true_features Ground truth binary targets.
 * @param grad_output Upstream gradient of loss with respect to output.
 * @param reduction Reduction mode applied in forward pass.
 * @return Gradient ndarray with respect to in_features matching in_features.shape.
 * @throw std::invalid_argument If shapes cannot broadcast together.
 */
template <typename T>
ndarray<T> bce_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    Reduction reduction
) {
    // Step 1: Compute broadcast shape and reduction scale factor
    size_t ndim_p = in_features.shape.size();
    size_t ndim_t = true_features.shape.size();
    size_t max_ndim = std::max(ndim_p, ndim_t);
    std::vector<size_t> padded_p = shape_utils::pad_shape(in_features.shape, max_ndim);
    std::vector<size_t> padded_t = shape_utils::pad_shape(true_features.shape, max_ndim);
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(padded_p, padded_t);
    size_t total_elements = shape_utils::num_elements(out_shape);

    T scale = static_cast<T>(1);
    if (reduction == Reduction::Mean) {
        scale = total_elements > 0 ? (static_cast<T>(1) / static_cast<T>(total_elements)) : static_cast<T>(0);
    }

    // Step 2: Evaluate analytical gradient
    bool grad_is_scalar = grad_output.isEmpty() || grad_output.size() == 1;
    constexpr T eps = static_cast<T>(1e-12);
    if (grad_is_scalar) {
        T scalar_grad = !grad_output.isEmpty() ? grad_output[0] : static_cast<T>(1);
        T total_scale = scale * scalar_grad;
        ndarray<T> full_grad = kernels::for_each_binary(
            in_features, true_features, [total_scale, eps](T p, T y) {
                T denom = std::max(p * (static_cast<T>(1) - p), eps);
                return total_scale * (p - y) / denom;
            }
        );
        // Step 3: Unbroadcast gradient back to in_features operand shape
        return shape_utils::unbroadcast(full_grad, in_features.shape);
    }

    ndarray<T> diff = kernels::for_each_binary(
        in_features, true_features, [scale, eps](T p, T y) {
            T denom = std::max(p * (static_cast<T>(1) - p), eps);
            return scale * (p - y) / denom;
        }
    );
    ndarray<T> full_grad = kernels::for_each_binary(
        diff, grad_output, [](T d, T g) {
            return d * g;
        }
    );
    // Step 3: Unbroadcast gradient back to in_features operand shape
    return shape_utils::unbroadcast(full_grad, in_features.shape);
}

/**
 * @brief Computes backward gradient of BCE loss accepting string reduction mode.
 *
 * @param in_features Input predicted probabilities.
 * @param true_features Ground truth binary targets.
 * @param grad_output Upstream gradient of loss with respect to output.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Gradient ndarray with respect to in_features.
 * @throw std::invalid_argument If reduction is unrecognized or shapes cannot broadcast.
 */
template <typename T>
ndarray<T> bce_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    const std::string& reduction
) {
    return bce_loss_backward(in_features, true_features, grad_output, parse_reduction(reduction));
}

// Explicit template instantiations
template ndarray<float> apply_reduction<float>(const ndarray<float>&, Reduction);
template ndarray<float> mse_loss<float>(const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> mse_loss<float>(const ndarray<float>&, const ndarray<float>&, const std::string&);
template ndarray<float> mse_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> mse_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, const std::string&);
template class MSELoss<float>;

template ndarray<double> apply_reduction<double>(const ndarray<double>&, Reduction);

template ndarray<double> mse_loss<double>(const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> mse_loss<double>(const ndarray<double>&, const ndarray<double>&, const std::string&);
template ndarray<double> mse_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> mse_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, const std::string&);
template class MSELoss<double>;

template ndarray<float> BCELoss<float>(const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> BCELoss<float>(const ndarray<float>&, const ndarray<float>&, const std::string&);
template ndarray<float> bce_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> bce_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, const std::string&);

template ndarray<double> BCELoss<double>(const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> BCELoss<double>(const ndarray<double>&, const ndarray<double>&, const std::string&);
template ndarray<double> bce_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> bce_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, const std::string&);

}  // namespace nn
}  // namespace gotorch
