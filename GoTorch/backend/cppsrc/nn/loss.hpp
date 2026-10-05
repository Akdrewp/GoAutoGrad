#ifndef GOTORCH_NN_LOSS_HPP
#define GOTORCH_NN_LOSS_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "gotorch/ndarray.hpp"
#include "gotorch/shape_utils.hpp"

namespace gotorch {
namespace nn {

/**
 * @brief Reduction mode applied to the loss output.
 */
enum class Reduction {
    Mean,  ///< Averages the squared loss elements across all dimensions.
    Sum,   ///< Sums the squared loss elements across all dimensions.
    None   ///< Does not reduce; preserves element-wise broadcast shape.
};

/**
 * @brief Parses reduction mode from string.
 *
 * @param reduction_str String name ("mean", "sum", "none").
 * @return Parsed Reduction enum value.
 * @throws std::invalid_argument If reduction_str is unrecognized.
 */
Reduction parse_reduction(const std::string& reduction_str);

/**
 * @brief Converts Reduction enum to string representation.
 *
 * @param reduction Reduction enum value.
 * @return String representation ("mean", "sum", "none").
 */
std::string reduction_to_string(Reduction reduction);

/**
 * @brief Applies reduction operation to an unreduced loss tensor.
 *
 * @param loss Unreduced loss ndarray.
 * @param reduction Reduction mode (Mean, Sum, None). Defaults to Mean.
 * @return Reduced loss ndarray (scalar for Mean/Sum, matching shape for None).
 */
template <typename T = float>
ndarray<T> apply_reduction(
    const ndarray<T>& loss,
    Reduction reduction = Reduction::Mean
);

/**
 * @brief Computes Mean Squared Error (MSE) loss between prediction and target.
 *
 * Measures the element-wise squared error: (prediction - target)^2, reduced
 * according to the specified mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values with broadcast-compatible shape.
 * @param reduction Reduction mode (Mean, Sum, None). Defaults to Mean.
 * @return NDArray containing the computed loss (scalar for Mean/Sum, matching shape for None).
 * @throws std::invalid_argument If shapes cannot be broadcast together.
 */
template <typename T = float>
ndarray<T> mse_loss(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    Reduction reduction = Reduction::Mean
);

/**
 * @brief Overload of mse_loss accepting string reduction mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return NDArray containing the computed loss.
 * @throws std::invalid_argument If reduction string is unrecognized or shapes cannot be broadcast.
 */
template <typename T = float>
ndarray<T> mse_loss(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const std::string& reduction
);

/**
 * @brief Computes backward gradient of MSE loss with respect to prediction.
 *
 * Computes:
 * - None: 2 * (prediction - target) * grad_output
 * - Sum:  2 * (prediction - target) * grad_output
 * - Mean: (2 / N) * (prediction - target) * grad_output
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param grad_output Gradient of loss with respect to output. If empty, defaults to 1.
 * @param reduction Reduction mode used during the forward pass.
 * @return Gradient ndarray with respect to prediction.
 * @throws std::invalid_argument If shapes cannot be broadcast or grad_output shape mismatches.
 */
template <typename T = float>
ndarray<T> mse_loss_backward(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const ndarray<T>& grad_output = ndarray<T>(),
    Reduction reduction = Reduction::Mean
);

/**
 * @brief Overload of mse_loss_backward accepting string reduction mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Gradient ndarray with respect to prediction.
 * @throws std::invalid_argument If reduction string is unrecognized or shapes cannot be broadcast.
 */
template <typename T = float>
ndarray<T> mse_loss_backward(
    const ndarray<T>& prediction,
    const ndarray<T>& target,
    const ndarray<T>& grad_output,
    const std::string& reduction
);

/**
 * @brief Class wrapper for Mean Squared Error loss.
 */
template <typename T = float>
class MSELoss {
public:
    Reduction reduction;

    /**
     * @brief Constructs an MSELoss module.
     *
     * @param reduction Mode of reduction. Defaults to Mean.
     */
    explicit MSELoss(Reduction reduction = Reduction::Mean) : reduction(reduction) {}

    /**
     * @brief Constructs an MSELoss module from a string mode.
     *
     * @param reduction_str String mode ("mean", "sum", "none").
     * @throws std::invalid_argument If reduction_str is unrecognized.
     */
    explicit MSELoss(const std::string& reduction_str) : reduction(parse_reduction(reduction_str)) {}

    /**
     * @brief Evaluates the forward MSE loss via function call operator.
     *
     * @param prediction Input predicted values.
     * @param target Ground truth target values.
     * @return Computed MSE loss ndarray.
     * @throws std::invalid_argument If shapes cannot be broadcast together.
     */
    ndarray<T> operator()(const ndarray<T>& prediction, const ndarray<T>& target) const {
        return mse_loss(prediction, target, reduction);
    }

    /**
     * @brief Evaluates the forward MSE loss.
     *
     * @param prediction Input predicted values.
     * @param target Ground truth target values.
     * @return Computed MSE loss ndarray.
     * @throws std::invalid_argument If shapes cannot be broadcast together.
     */
    ndarray<T> forward(const ndarray<T>& prediction, const ndarray<T>& target) const {
        return mse_loss(prediction, target, reduction);
    }

    /**
     * @brief Evaluates the backward MSE gradient with respect to prediction.
     *
     * @param prediction Input predicted values.
     * @param target Ground truth target values.
     * @param grad_output Gradient of loss with respect to output. Defaults to scalar 1.
     * @return Gradient ndarray with respect to prediction.
     * @throws std::invalid_argument If shapes cannot be broadcast or grad_output shape mismatches.
     */
    ndarray<T> backward(
        const ndarray<T>& prediction,
        const ndarray<T>& target,
        const ndarray<T>& grad_output = ndarray<T>()
    ) const {
        return mse_loss_backward(prediction, target, grad_output, reduction);
    }
};

/**
 * @brief Computes Binary Cross Entropy (BCE) loss between prediction and target.
 *
 * @param in_features Input predicted values.
 * @param true_features Ground truth target values with broadcast-compatible shape.
 * @param reduction Reduction mode (Mean, Sum, None). Defaults to Mean.
 * @return NDArray containing the computed loss (scalar for Mean/Sum, matching shape for None).
 * @throws std::invalid_argument If shapes cannot be broadcast together.
 */
template <typename T = float>
ndarray<T> BCELoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    Reduction reduction = Reduction::Mean
);

/**
 * @brief Overload of BCELoss accepting string reduction mode.
 *
 * @param in_features Input predicted values.
 * @param true_features Ground truth target values.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return NDArray containing the computed loss.
 * @throws std::invalid_argument If reduction string is unrecognized or shapes cannot be broadcast.
 */
template <typename T = float>
ndarray<T> BCELoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const std::string& reduction
);

/**
 * @brief Computes backward gradient of BCE loss with respect to in_features.
 *
 * @param in_features Input predicted values.
 * @param true_features Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction Reduction mode used during the forward pass.
 * @return Gradient ndarray with respect to in_features.
 * @throws std::invalid_argument If shapes cannot be broadcast or grad_output shape mismatches.
 */
template <typename T = float>
ndarray<T> bce_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output = ndarray<T>(),
    Reduction reduction = Reduction::Mean
);

/**
 * @brief Overload of bce_loss_backward accepting string reduction mode.
 *
 * @param in_features Input predicted values.
 * @param true_features Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Gradient ndarray with respect to in_features.
 * @throws std::invalid_argument If reduction string is unrecognized or shapes cannot be broadcast.
 */
template <typename T = float>
ndarray<T> bce_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    const std::string& reduction
);

}  // namespace nn
}  // namespace gotorch

#endif  // GOTORCH_NN_LOSS_HPP
