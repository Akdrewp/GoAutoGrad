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
 * 1. Validate matching operand shapes and compute squared differences.
 * 2. Reduce squared error elements according to configured reduction mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values with identical shape.
 * @param reduction Reduction mode (Mean, Sum, None).
 * @return Computed loss ndarray (scalar for Mean/Sum, matching shape for None).
 * @throw std::invalid_argument If shapes do not match.
 */
template <typename T>
ndarray<T> mse_loss(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    Reduction reduction
) {
    // 1. Evaluate element-wise squared difference and reduction via Loss base
    return Loss<T>(reduction).evaluate_forward(
        prediction, target, [](T p, T t) {
            T diff = p - t;
            return diff * diff;
        }
    );
}

/**
 * @brief Computes MSE loss accepting string reduction mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Computed loss ndarray.
 * @throw std::invalid_argument If reduction string is unrecognized or shapes do not match.
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
 *
 * 1. Validate operand and gradient shapes.
 * 2. Evaluate analytical gradient scaled by reduction factor.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction Reduction mode applied in forward pass.
 * @return Gradient ndarray with respect to prediction matching prediction.shape.
 * @throw std::invalid_argument If shapes do not match or grad_output shape mismatches.
 */
template <typename T>
ndarray<T> mse_loss_backward(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const ndarray<T>& grad_output,
    Reduction reduction
) {
    // 1. Evaluate analytical gradient via Loss base with scaling factor 2
    return Loss<T>(reduction).evaluate_backward(
        prediction, target, grad_output, [](T p, T t) {
            return p - t;
        }, static_cast<T>(2)
    );
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
 * 1. Validate matching operand shapes and compute element-wise binary cross entropy.
 * 2. Apply requested reduction to element-wise losses.
 *
 * @param in_features Input predicted probabilities.
 * @param true_features Ground truth binary targets (0 or 1).
 * @param reduction Reduction mode (Mean, Sum, None).
 * @return Computed loss ndarray (scalar for Mean/Sum, matching shape for None).
 * @throw std::invalid_argument If shapes do not match.
 */
template <typename T>
ndarray<T> BCELoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    Reduction reduction
) {
    // 1. Evaluate binary cross entropy and reduce via Loss base
    return Loss<T>(reduction).evaluate_forward(
        in_features, true_features, [](T p, T y) {
            return -((y * std::log(p)) + ((static_cast<T>(1) - y) * std::log(static_cast<T>(1) - p)));
        }
    );
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
 * 1. Validate operand and gradient shapes.
 * 2. Evaluate analytical gradient: (p - y) / (p * (1 - p)) * scale * grad_output.
 *
 * @param in_features Input predicted probabilities.
 * @param true_features Ground truth binary targets.
 * @param grad_output Upstream gradient of loss with respect to output.
 * @param reduction Reduction mode applied in forward pass.
 * @return Gradient ndarray with respect to in_features matching in_features.shape.
 * @throw std::invalid_argument If shapes do not match or grad_output shape mismatches.
 */
template <typename T>
ndarray<T> bce_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    Reduction reduction
) {
    // 1. Evaluate analytical gradient via Loss base with epsilon safeguard
    constexpr T eps = static_cast<T>(1e-12);
    return Loss<T>(reduction).evaluate_backward(
        in_features, true_features, grad_output, [eps](T p, T y) {
            T denom = std::max(p * (static_cast<T>(1) - p), eps);
            return (p - y) / denom;
        }
    );
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

/**
 * @brief Computes Binary Cross Entropy with Logits (BCEWithLogits) loss between logits and target.
 *
 * Combines a sigmoid activation and binary cross entropy into a single
 * numerically stable operation using the log-sum-exp formulation.
 *
 * 1. Validate matching operand shapes and compute element-wise numerically stable loss.
 * 2. Apply requested reduction to loss elements.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth binary targets (0 or 1).
 * @param reduction Reduction mode (Mean, Sum, None).
 * @return Computed loss ndarray (scalar for Mean/Sum, matching shape for None).
 * @throw std::invalid_argument If shapes do not match.
 */
template <typename T>
ndarray<T> BCEWithLogitsLoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    Reduction reduction
) {
    // 1. Evaluate numerically stable logits cross entropy and reduce via Loss base
    return Loss<T>(reduction).evaluate_forward(
        in_features, true_features, [](T x, T y) {
            T max_val = std::max(x, static_cast<T>(0));
            T abs_x = std::abs(x);
            return max_val - (x * y) + std::log1p(std::exp(-abs_x));
        }
    );
}

/**
 * @brief Computes BCEWithLogits loss accepting string reduction mode.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth binary targets.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Computed loss ndarray.
 * @throw std::invalid_argument If reduction string is unrecognized or shapes cannot broadcast.
 */
template <typename T>
ndarray<T> BCEWithLogitsLoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const std::string& reduction
) {
    return BCEWithLogitsLoss(in_features, true_features, parse_reduction(reduction));
}

/**
 * @brief Computes backward gradient of BCEWithLogits loss with respect to in_features.
 *
 * Evaluates the analytical gradient: (sigmoid(x) - y) * scale * grad_output.
 *
 * 1. Validate operand and gradient shapes.
 * 2. Evaluate analytical gradient elements using numerically stable sigmoid.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth binary targets.
 * @param grad_output Upstream gradient of loss with respect to output.
 * @param reduction Reduction mode applied in forward pass.
 * @return Gradient ndarray with respect to in_features matching in_features.shape.
 * @throw std::invalid_argument If shapes do not match or grad_output shape mismatches.
 */
template <typename T>
ndarray<T> bce_with_logits_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    Reduction reduction
) {
    // 1. Evaluate analytical gradient via Loss base using numerically stable sigmoid
    auto sigmoid_fn = [](T x) -> T {
        if (x >= static_cast<T>(0)) {
            return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
        }
        T exp_x = std::exp(x);
        return exp_x / (static_cast<T>(1) + exp_x);
    };
    return Loss<T>(reduction).evaluate_backward(
        in_features, true_features, grad_output, [sigmoid_fn](T x, T y) {
            return sigmoid_fn(x) - y;
        }
    );
}

/**
 * @brief Computes backward gradient of BCEWithLogits loss accepting string reduction mode.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth binary targets.
 * @param grad_output Upstream gradient of loss with respect to output.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Gradient ndarray with respect to in_features.
 * @throw std::invalid_argument If reduction is unrecognized or shapes cannot broadcast.
 */
template <typename T>
ndarray<T> bce_with_logits_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    const std::string& reduction
) {
    return bce_with_logits_loss_backward(in_features, true_features, grad_output, parse_reduction(reduction));
}

// Explicit template instantiations
template class Loss<float>;
template ndarray<float> apply_reduction<float>(const ndarray<float>&, Reduction);
template ndarray<float> mse_loss<float>(const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> mse_loss<float>(const ndarray<float>&, const ndarray<float>&, const std::string&);
template ndarray<float> mse_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> mse_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, const std::string&);
template class MSELoss<float>;

template class Loss<double>;
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

template ndarray<float> BCEWithLogitsLoss<float>(const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> BCEWithLogitsLoss<float>(const ndarray<float>&, const ndarray<float>&, const std::string&);
template ndarray<float> bce_with_logits_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, Reduction);
template ndarray<float> bce_with_logits_loss_backward<float>(const ndarray<float>&, const ndarray<float>&, const ndarray<float>&, const std::string&);

template ndarray<double> BCEWithLogitsLoss<double>(const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> BCEWithLogitsLoss<double>(const ndarray<double>&, const ndarray<double>&, const std::string&);
template ndarray<double> bce_with_logits_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, Reduction);
template ndarray<double> bce_with_logits_loss_backward<double>(const ndarray<double>&, const ndarray<double>&, const ndarray<double>&, const std::string&);

}  // namespace nn
}  // namespace gotorch
