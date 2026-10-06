#ifndef GOTORCH_NN_LOSS_HPP
#define GOTORCH_NN_LOSS_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "gotorch/kernels.hpp"
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
 * @brief Base class for loss functions providing common shape validation and reduction handling.
 */
template <typename T = float>
class Loss {
public:
    Reduction reduction;

    /**
     * @brief Constructs a Loss criterion with the specified reduction mode.
     *
     * @param reduction Reduction mode (Mean, Sum, None). Defaults to Mean.
     */
    explicit Loss(Reduction reduction = Reduction::Mean) : reduction(reduction) {}

    /**
     * @brief Constructs a Loss criterion from a reduction string name.
     *
     * @param reduction_str String name of reduction ("mean", "sum", "none").
     * @throw std::invalid_argument If reduction string is unrecognized.
     */
    explicit Loss(const std::string& reduction_str) : reduction(parse_reduction(reduction_str)) {}

    /**
     * @brief Destructs the Loss base instance.
     */
    virtual ~Loss() = default;

    /**
     * @brief Validates input and target shapes match exactly.
     *
     * @param input Prediction/input tensor.
     * @param target Ground truth target tensor.
     * @throw std::invalid_argument If shapes do not match.
     */
    static void validate_shapes(const ndarray<T>& input, const ndarray<T>& target) {
        if (input.shape != target.shape) {
            throw std::invalid_argument(
                "Input and target shapes must match; broadcasting is not supported in loss functions."
            );
        }
    }

    /**
     * @brief Validates grad_output shape compatibility.
     *
     * Scalar or empty gradients are universally compatible. Non-scalar gradients
     * must match the unreduced input tensor shape.
     *
     * @param input Input tensor.
     * @param grad_output Upstream gradient tensor.
     * @throw std::invalid_argument If grad_output shape is incompatible.
     */
    static void validate_grad_output(const ndarray<T>& input, const ndarray<T>& grad_output) {
        // Scalar or empty gradients are universally compatible
        if (grad_output.isEmpty() || grad_output.size() == 1) {
            return;
        }
        if (grad_output.shape != input.shape) {
            throw std::invalid_argument(
                "grad_output shape must match prediction shape for unreduced loss."
            );
        }
    }

    /**
     * @brief Validates operands, evaluates element-wise loss, and applies reduction.
     *
     * 1. Validate that input and target shapes match exactly.
     * 2. Compute element-wise unreduced loss via binary kernel.
     * 3. Apply reduction mode to unreduced loss elements.
     *
     * @tparam BinaryOp Binary callable computing element loss (T, T) -> T.
     * @param input Input predicted values.
     * @param target Ground truth target values.
     * @param op Element-wise loss operation.
     * @return Reduced loss ndarray.
     * @throw std::invalid_argument If shapes mismatch.
     */
    template <typename BinaryOp>
    ndarray<T> evaluate_forward(
        const ndarray<T>& input,
        const ndarray<T>& target,
        BinaryOp&& op
    ) const {
        // 1. Verify shape alignment
        validate_shapes(input, target);
        // 2. Compute element-wise loss
        ndarray<T> unreduced = kernels::for_each_binary(input, target, std::forward<BinaryOp>(op));
        // 3. Apply requested reduction
        return apply_reduction(unreduced, reduction);
    }

    /**
     * @brief Validates operands and evaluates backward gradient with reduction scaling.
     *
     * 1. Validate that input and target shapes match exactly.
     * 2. Validate upstream grad_output shape.
     * 3. Compute analytical gradient elements scaled by reduction mode and grad_output.
     *
     * @tparam GradOp Binary callable computing unscaled analytical gradient (T, T) -> T.
     * @param input Input predicted values.
     * @param target Ground truth target values.
     * @param grad_output Upstream gradient of loss with respect to output.
     * @param op Element-wise gradient operation.
     * @param scale_factor Additional scaling factor (e.g. 2 for MSE). Defaults to 1.
     * @return Gradient ndarray with respect to input.
     * @throw std::invalid_argument If shapes mismatch or grad_output shape is invalid.
     */
    template <typename GradOp>
    ndarray<T> evaluate_backward(
        const ndarray<T>& input,
        const ndarray<T>& target,
        const ndarray<T>& grad_output,
        GradOp&& op,
        T scale_factor = static_cast<T>(1)
    ) const {
        // 1. Verify shape alignment
        validate_shapes(input, target);
        // 2. Verify grad_output shape
        validate_grad_output(input, grad_output);

        // 3. Evaluate analytical gradient elements
        size_t total_elements = input.size();
        T scale = scale_factor;
        if (reduction == Reduction::Mean) {
            scale = total_elements > 0 ? (scale_factor / static_cast<T>(total_elements)) : static_cast<T>(0);
        }

        bool grad_is_scalar = grad_output.isEmpty() || grad_output.size() == 1;
        if (grad_is_scalar) {
            T scalar_grad = !grad_output.isEmpty() ? grad_output[0] : static_cast<T>(1);
            T total_scale = scale * scalar_grad;
            return kernels::for_each_binary(
                input, target, [total_scale, &op](T in_val, T tgt_val) {
                    return total_scale * op(in_val, tgt_val);
                }
            );
        }

        ndarray<T> diff = kernels::for_each_binary(
            input, target, [scale, &op](T in_val, T tgt_val) {
                return scale * op(in_val, tgt_val);
            }
        );
        return kernels::for_each_binary(
            diff, grad_output, [](T d, T g) {
                return d * g;
            }
        );
    }
};

/**
 * @brief Computes Mean Squared Error (MSE) loss between prediction and target.
 *
 * Measures the element-wise squared error: (prediction - target)^2, reduced
 * according to the specified mode.
 *
 * @param prediction Input predicted values.
 * @param target Ground truth target values with identical shape.
 * @param reduction Reduction mode (Mean, Sum, None). Defaults to Mean.
 * @return NDArray containing the computed loss (scalar for Mean/Sum, matching shape for None).
 * @throws std::invalid_argument If shapes do not match.
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
 * @throws std::invalid_argument If reduction string is unrecognized or shapes do not match.
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
 * @throws std::invalid_argument If shapes do not match or grad_output shape mismatches.
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
 * @throws std::invalid_argument If reduction string is unrecognized or shapes do not match.
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
class MSELoss : public Loss<T> {
public:
    using Loss<T>::Loss;

    /**
     * @brief Evaluates the forward MSE loss via function call operator.
     *
     * @param prediction Input predicted values.
     * @param target Ground truth target values.
     * @return Computed MSE loss ndarray.
     * @throws std::invalid_argument If shapes do not match.
     */
    ndarray<T> operator()(const ndarray<T>& prediction, const ndarray<T>& target) const {
        return mse_loss(prediction, target, this->reduction);
    }

    /**
     * @brief Evaluates the forward MSE loss.
     *
     * @param prediction Input predicted values.
     * @param target Ground truth target values.
     * @return Computed MSE loss ndarray.
     * @throws std::invalid_argument If shapes do not match.
     */
    ndarray<T> forward(const ndarray<T>& prediction, const ndarray<T>& target) const {
        return mse_loss(prediction, target, this->reduction);
    }

    /**
     * @brief Evaluates the backward MSE gradient with respect to prediction.
     *
     * @param prediction Input predicted values.
     * @param target Ground truth target values.
     * @param grad_output Gradient of loss with respect to output. Defaults to scalar 1.
     * @return Gradient ndarray with respect to prediction.
     * @throws std::invalid_argument If shapes do not match or grad_output shape mismatches.
     */
    ndarray<T> backward(
        const ndarray<T>& prediction,
        const ndarray<T>& target,
        const ndarray<T>& grad_output = ndarray<T>()
    ) const {
        return mse_loss_backward(prediction, target, grad_output, this->reduction);
    }
};

/**
 * @brief Computes Binary Cross Entropy (BCE) loss between prediction and target.
 *
 * @param in_features Input predicted values.
 * @param true_features Ground truth target values with identical shape.
 * @param reduction Reduction mode (Mean, Sum, None). Defaults to Mean.
 * @return NDArray containing the computed loss (scalar for Mean/Sum, matching shape for None).
 * @throws std::invalid_argument If shapes do not match.
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
 * @throws std::invalid_argument If reduction string is unrecognized or shapes do not match.
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
 * @throws std::invalid_argument If shapes do not match or grad_output shape mismatches.
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
 * @throws std::invalid_argument If reduction string is unrecognized or shapes do not match.
 */
template <typename T = float>
ndarray<T> bce_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    const std::string& reduction
);

/**
 * @brief Computes Binary Cross Entropy with Logits (BCEWithLogits) loss between logits and target.
 *
 * Combines a sigmoid activation and binary cross entropy into a single
 * numerically stable operation using the log-sum-exp formulation.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth target values with identical shape.
 * @param reduction Reduction mode (Mean, Sum, None). Defaults to Mean.
 * @return NDArray containing the computed loss (scalar for Mean/Sum, matching shape for None).
 * @throws std::invalid_argument If shapes do not match.
 */
template <typename T = float>
ndarray<T> BCEWithLogitsLoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    Reduction reduction = Reduction::Mean
);

/**
 * @brief Overload of BCEWithLogitsLoss accepting string reduction mode.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth target values.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return NDArray containing the computed loss.
 * @throws std::invalid_argument If reduction string is unrecognized or shapes do not match.
 */
template <typename T = float>
ndarray<T> BCEWithLogitsLoss(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const std::string& reduction
);

/**
 * @brief Computes backward gradient of BCEWithLogits loss with respect to in_features.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction Reduction mode used during the forward pass.
 * @return Gradient ndarray with respect to in_features.
 * @throws std::invalid_argument If shapes do not match or grad_output shape mismatches.
 */
template <typename T = float>
ndarray<T> bce_with_logits_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output = ndarray<T>(),
    Reduction reduction = Reduction::Mean
);

/**
 * @brief Overload of bce_with_logits_loss_backward accepting string reduction mode.
 *
 * @param in_features Input logits.
 * @param true_features Ground truth target values.
 * @param grad_output Gradient of loss with respect to output.
 * @param reduction String reduction mode ("mean", "sum", "none").
 * @return Gradient ndarray with respect to in_features.
 * @throws std::invalid_argument If reduction string is unrecognized or shapes do not match.
 */
template <typename T = float>
ndarray<T> bce_with_logits_loss_backward(
    const ndarray<T>& in_features,
    const ndarray<T>& true_features,
    const ndarray<T>& grad_output,
    const std::string& reduction
);

}  // namespace nn
}  // namespace gotorch

#endif  // GOTORCH_NN_LOSS_HPP
