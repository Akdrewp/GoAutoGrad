#ifndef GOTORCH_NDARRAY_HPP
#define GOTORCH_NDARRAY_HPP

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "gotorch/shape_utils.hpp"

template <typename T = float>
struct Storage {
    std::vector<T> data;

    Storage() = default;
    explicit Storage(size_t size) : data(size) {}
    Storage(size_t size, const T& init_val) : data(size, init_val) {}
    explicit Storage(std::vector<T> vec) : data(std::move(vec)) {}
    Storage(const T* ptr, size_t size) : data(ptr, ptr + size) {}
};

template <typename T = float>
class ndarray {
public:
    std::shared_ptr<Storage<T>> storage;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset;

    /**
     * @brief Computes default row-major (C-contiguous) strides for a given shape.
     */
    static std::vector<size_t> default_strides(const std::vector<size_t>& shape) {
        return gotorch::shape_utils::default_strides(shape);
    }

    /**
     * @brief Calculates total number of elements in a given shape.
     */
    static size_t num_elements(const std::vector<size_t>& shape) {
        return gotorch::shape_utils::num_elements(shape);
    }

    /**
     * @brief Creates a deep copy of another ndarray with independent storage.
     */
    void CreateDeepCopy(const ndarray& other);

    /**
     * @brief Allocates and initializes ndarray storage from raw pointer and shape.
     */
    void CreateNDArray(
        const T* array,
        const std::vector<size_t>& shape_dims,
        const std::vector<size_t>& custom_strides = {},
        size_t off = 0
    );

    /**
     * @brief Default constructor creating an empty ndarray.
     */
    ndarray()
        : storage(std::make_shared<Storage<T>>()), shape({0}), strides({1}), offset(0) {}

    /**
     * @brief Constructor creating an ndarray by deep copying an existing ndarray.
     */
    ndarray(const ndarray& copyArray)
        : offset(0) {
        CreateDeepCopy(copyArray);
    }

    /**
     * @brief Constructor creating an ndarray from pointer to data array and shape.
     */
    ndarray(
        const T* array,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides = {},
        size_t offset = 0
    ) {
        CreateNDArray(array, shape, strides, offset);
    }

    /**
     * @brief Constructor creating an ndarray from std::vector and shape.
     */
    ndarray(
        const std::vector<T>& vec,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides = {},
        size_t offset = 0
    );

    /**
     * @brief Constructor creating an ndarray with existing shared storage.
     */
    ndarray(
        std::shared_ptr<Storage<T>> storage,
        std::vector<size_t> shape,
        std::vector<size_t> strides,
        size_t offset = 0
    ) : storage(std::move(storage)),
        shape(std::move(shape)),
        strides(std::move(strides)),
        offset(offset) {}

    /**
     * @brief Creates an independent deep clone of this ndarray.
     */
    ndarray clone() const {
        return ndarray(*this);
    }

    /**
     * @brief Creates a zero-copy sliced view sharing the same storage.
     */
    ndarray view(
        std::vector<size_t> new_shape,
        std::vector<size_t> new_strides,
        size_t new_offset = 0
    ) const {
        return ndarray(storage, std::move(new_shape), std::move(new_strides), new_offset);
    }

    /**
     * @brief Creates a zero-copy transposed view by swapping two dimensions.
     */
    ndarray transpose(size_t dim0 = 0, size_t dim1 = 1) const;

    /**
     * @brief Reshapes the ndarray to a new shape if contiguous.
     */
    ndarray reshape(const std::vector<size_t>& new_shape) const;

    /**
     * @brief Checks if memory layout matches default row-major contiguous strides.
     */
    bool is_contiguous() const {
        return strides == default_strides(shape) && offset == 0;
    }

    /**
     * @brief Checks whether the ndarray contains zero elements based on its shape.
     */
    bool isEmpty() const {
        return size() == 0;
    }

    /**
     * @brief Total number of elements represented by this view.
     */
    size_t size() const {
        return num_elements(shape);
    }

    /**
     * @brief Maps multidimensional indices to flat buffer offset.
     */
    size_t index_to_offset(const std::vector<size_t>& indices) const {
        return gotorch::shape_utils::index_to_offset(indices, shape, strides, offset);
    }

    /**
     * @brief Returns an array of ones with identical shape and standard strides.
     */
    ndarray ones_like() const {
        return ndarray(std::vector<T>(size(), static_cast<T>(1)), shape);
    }

    /**
     * @brief Returns an array of zeros with identical shape and standard strides.
     */
    ndarray zeros_like() const {
        return ndarray(std::vector<T>(size(), static_cast<T>(0)), shape);
    }

    /**
     * @brief Direct pointer access to underlying buffer (at offset).
     */
    T* data() {
        return storage ? storage->data.data() + offset : nullptr;
    }

    const T* data() const {
        return storage ? storage->data.data() + offset : nullptr;
    }

    /**
     * @brief Access element via multidimensional indices.
     */
    const T& operator()(const std::vector<size_t>& indices) const {
        return storage->data[index_to_offset(indices)];
    }

    T& operator()(const std::vector<size_t>& indices) {
        return storage->data[index_to_offset(indices)];
    }

    /**
     * @brief Access element via 1D buffer index.
     */
    const T& operator[](size_t idx) const {
        return storage->data[offset + idx];
    }

    T& operator[](size_t idx) {
        return storage->data[offset + idx];
    }

    // Mathematical Operations (defined in kernels.cpp)
    ndarray add(const ndarray& addend) const;
    ndarray operator+(const ndarray& addend) const {
        return add(addend);
    }

    ndarray sub(const ndarray& subtrahend) const;
    ndarray operator-(const ndarray& subtrahend) const {
        return sub(subtrahend);
    }

    ndarray matmul(const ndarray& other) const;
    ndarray relu() const;
    ndarray relu_backward(const ndarray& grad_output) const;
    ndarray tanh() const;
    ndarray tanh_backward(const ndarray& grad_output) const;
    ndarray sigmoid() const;
    ndarray sigmoid_backward(const ndarray& grad_output) const;
    ndarray leaky_relu(T alpha = static_cast<T>(0.01)) const;
    ndarray leaky_relu_backward(const ndarray& grad_output, T alpha = static_cast<T>(0.01)) const;
    ndarray sum(int dim, bool keepdim = false) const;
    ndarray sum() const;
    ndarray unbroadcast(const std::vector<size_t>& target_shape) const;
};

using NDArray = ndarray<float>;

#endif  // GOTORCH_NDARRAY_HPP
