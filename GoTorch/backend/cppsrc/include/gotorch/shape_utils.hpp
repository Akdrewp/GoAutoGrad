#ifndef GOTORCH_SHAPE_UTILS_HPP
#define GOTORCH_SHAPE_UTILS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declaration for ndarray template in global namespace
template <typename T>
class ndarray;

namespace gotorch {
namespace shape_utils {

/**
 * @brief Computes total number of elements for a given shape vector.
 * Empty shape represents a 0-D scalar and has 1 element.
 *
 * @param shape Vector of dimension sizes.
 * @return Total product of dimension sizes.
 */
size_t num_elements(const std::vector<size_t>& shape);

/**
 * @brief Computes standard row-major (C-contiguous) strides for a given shape.
 *
 * In a standard C-contiguous layout, the innermost dimension has a stride of 1,
 * and each preceding dimension's stride is the product of all following dimension sizes:
 *   stride[i] = stride[i + 1] * shape[i + 1]
 *
 * @param shape Shape dimensions vector.
 * @return Row-major strides vector for each dimension.
 */
std::vector<size_t> default_strides(const std::vector<size_t>& shape);

/**
 * @brief Maps multidimensional coordinate indices to a flat linear memory buffer offset.
 *
 * @param indices Coordinates in the ndarray.
 * @param shape Shape vector of the array.
 * @param strides Strides vector of the array.
 * @param base_offset Memory level base offset.
 * @return Flat memory offset.
 * @throws std::out_of_range If coordinate count does not match shape rank.
 */
size_t index_to_offset(
    const std::vector<size_t>& indices,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    size_t base_offset = 0
);

/**
 * @brief Computes flat linear buffer offset from multidimensional coordinates and strides.
 *
 * @param coord Coordinate vector.
 * @param strides Stride vector.
 * @return Flat buffer offset.
 */
size_t compute_coordinate_offset(
    const std::vector<size_t>& coord,
    const std::vector<size_t>& strides
);

/**
 * @brief Advances a coordinate vector by one step in row-major order.
 *
 * @param coord Coordinate vector to increment in-place.
 * @param shape Shape boundary vector.
 */
void advance_coordinate(
    std::vector<size_t>& coord,
    const std::vector<size_t>& shape
);

/**
 * @brief Left-pads a shape vector with 1s to match the target rank.
 *
 * @param shape Source shape vector.
 * @param target_rank Target rank to pad up to.
 * @return Padded shape vector of size target_rank.
 */
std::vector<size_t> pad_shape(
    const std::vector<size_t>& shape,
    size_t target_rank
);

/**
 * @brief Validates broadcast compatibility between two padded shapes and determines output shape.
 *
 * @param padded_a First padded shape vector.
 * @param padded_b Second padded shape vector.
 * @return Resulting broadcast output shape vector.
 * @throws std::invalid_argument If dimensions cannot be broadcast together.
 */
std::vector<size_t> broadcast_shapes(
    const std::vector<size_t>& padded_a,
    const std::vector<size_t>& padded_b
);

/**
 * @brief Computes effective broadcasted strides for an operand.
 * Broadcasted or padded dimensions receive a stride of 0.
 *
 * @param shape Original shape vector.
 * @param strides Original strides vector.
 * @param target_rank Target broadcast rank.
 * @return Effective strides vector of size target_rank.
 */
std::vector<size_t> compute_broadcast_strides(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    size_t target_rank
);

/**
 * @brief Normalizes a reduction dimension index, supporting negative indexing.
 *
 * @param dim The dimension index (can be negative).
 * @param rank The rank of the array.
 * @return Normalized non-negative axis index in [0, rank - 1].
 * @throws std::out_of_range If dim is not in [-rank, rank - 1].
 */
size_t normalize_axis(int dim, size_t rank);

/**
 * @brief Computes the output shape after reducing along a specified axis.
 *
 * @param shape Input shape vector.
 * @param axis Dimension axis to reduce.
 * @param keepdim Whether to retain the reduced dimension with size 1.
 * @return Reduced shape vector.
 */
std::vector<size_t> compute_reduced_shape(
    const std::vector<size_t>& shape,
    size_t axis,
    bool keepdim
);

/**
 * @brief Maps an input coordinate vector to its corresponding reduced output coordinate vector.
 *
 * @param in_coord Input coordinate vector.
 * @param axis The reduced axis.
 * @param keepdim Whether the reduced axis is retained as size 1.
 * @param out_coord Output coordinate buffer to populate.
 */
void map_reduced_coordinate(
    const std::vector<size_t>& in_coord,
    size_t axis,
    bool keepdim,
    std::vector<size_t>& out_coord
);

/**
 * @brief Sums an ndarray across broadcasted dimensions to match target_shape.
 *
 * @param in Input ndarray to reduce.
 * @param target_shape Target unbroadcast shape.
 * @return Reduced ndarray matching target_shape.
 */
template <typename T>
::ndarray<T> unbroadcast(const ::ndarray<T>& in, const std::vector<size_t>& target_shape);

}  // namespace shape_utils
}  // namespace gotorch

#endif  // GOTORCH_SHAPE_UTILS_HPP
