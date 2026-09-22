#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 *  struct Storage {
 *      std::vector<T> data;
 *  }
 */
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
     * @param shape Shape dimensions of the array.
     * @return Stride vector for each dimension.
     */
    static std::vector<size_t> default_strides(const std::vector<size_t>& shape);

    /**
     * @brief Calculates total number of elements in a given shape.
     * @param shape Shape dimensions.
     * @return Total element count.
     */
    static size_t num_elements(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * @brief Creates a deep copy of another ndarray with independent storage.
     * @param other The ndarray instance to deeply copy from.
     */
    void CreateDeepCopy(const ndarray& other);

    /**
     * @brief Allocates and initializes ndarray storage from raw pointer and shape.
     * @param array Pointer to contiguous or source array data.
     * @param shape_dims Shape dimensions of the ndarray.
     * @param custom_strides Optional custom strides for strided view.
     * @param off Memory offset within the buffer.
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
        : storage(std::make_shared<Storage<T>>()), shape({}), strides({}), offset(0) {}

    /**
     * @brief Constructor creating an ndarray by deep copying an existing ndarray.
     * @param copyArray ndarray to copy.
     */
    ndarray(const ndarray& copyArray)
        : offset(0) {
        CreateDeepCopy(copyArray);
    }

    /**
     * @brief Constructor creating an ndarray from pointer to data array and shape.
     * @param array Pointer to array data.
     * @param shape Shape dimensions of the array.
     * @param strides Strides for the array (defaults to row-major).
     * @param offset Memory level offset (defaults to 0).
     */
    ndarray(
        const T* array,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides = {},
        size_t offset = 0
    ) : offset(0) {
        CreateNDArray(array, shape, strides, offset);
    }

    /**
     * @brief Constructor creating an ndarray from std::vector and shape.
     * @param vec Vector of elements.
     * @param shape Shape dimensions of the array.
     * @param strides Strides for the array (defaults to row-major).
     * @param offset Memory level offset (defaults to 0).
     */
    ndarray(
        const std::vector<T>& vec,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides = {},
        size_t offset = 0
    );

    /**
     * @brief Constructor creating a zero-copy view referencing shared storage.
     * @param shared_storage Shared pointer to underlying data storage.
     * @param shape View shape dimensions.
     * @param strides View strides.
     * @param offset View memory offset.
     */
    ndarray(
        std::shared_ptr<Storage<T>> shared_storage,
        std::vector<size_t> shape,
        std::vector<size_t> strides,
        size_t offset = 0
    ) : storage(std::move(shared_storage)),
        shape(std::move(shape)),
        strides(std::move(strides)),
        offset(offset) {}

    ~ndarray() = default;

    /**
     * @brief Copy assignment operator (performs deep copy).
     * @param other ndarray to copy from.
     * @return Reference to self.
     */
    ndarray& operator=(const ndarray& other) {
        if (this != &other) {
            CreateDeepCopy(other);
        }
        return *this;
    }

    /**
     * @brief Move constructor.
     * @param other ndarray to move from.
     */
    ndarray(ndarray&& other) noexcept = default;

    /**
     * @brief Move assignment operator.
     * @param other ndarray to move from.
     * @return Reference to self.
     */
    ndarray& operator=(ndarray&& other) noexcept = default;

    /**
     * @brief Creates a zero-copy transposed view by swapping shape and strides.
     * @param dim0 First dimension to swap.
     * @param dim1 Second dimension to swap.
     * @return New ndarray viewing the exact same shared storage.
     */
    ndarray transpose(size_t dim0 = 0, size_t dim1 = 1) const;

    /**
     * @brief Performs element-wise addition with broadcasting.
     *
     * Shapes are padded to the left with 1s to match the maximum rank.
     * If corresponding dimensions do not match and neither is 1, a
     * std::invalid_argument exception is thrown.
     *
     * @param addend The array to add to this array.
     * @return New ndarray containing the broadcasted addition result.
     * @throws std::invalid_argument If shapes cannot be broadcasted together.
     */
    ndarray add(const ndarray& addend) const;

    /**
     * @brief Operator overload for element-wise addition with broadcasting.
     * @param addend The array to add to this array.
     * @return New ndarray containing the broadcasted addition result.
     */
    ndarray operator+(const ndarray& addend) const {
        return add(addend);
    }

    /**
     * @brief Performs 2D matrix multiplication: (M, K) @ (K, N) -> (M, N).
     * @param other The matrix to multiply with.
     * @return New ndarray containing the matrix multiplication result.
     * @throws std::invalid_argument If shapes are not 2D or inner dimensions mismatch.
     */
    ndarray matmul(const ndarray& other) const;

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
     * @brief Creates a zero-copy sliced view sharing the same storage.
     * @param new_shape New view shape.
     * @param new_strides New view strides.
     * @param new_offset New memory offset.
     * @return New ndarray viewing the shared storage.
     */
    ndarray view(
        std::vector<size_t> new_shape,
        std::vector<size_t> new_strides,
        size_t new_offset = 0
    ) const {
        return ndarray(storage, std::move(new_shape), std::move(new_strides), new_offset);
    }

    /**
     * @brief Maps multidimensional indices to flat buffer offset.
     * @param indices Coordinates in the ndarray.
     * @return Flat memory offset.
     */
    size_t index_to_offset(const std::vector<size_t>& indices) const;

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
     * @brief Total number of elements represented by this view.
     */
    size_t size() const {
        return num_elements(shape);
    }

    /**
     * @brief Access element via multidimensional indices.
     */
    const T& operator()(const std::vector<size_t>& indices) const {
        return storage->data[index_to_offset(indices)];
    }

    /**
     * @brief Access element via multidimensional indices (mutable).
     */
    T& operator()(const std::vector<size_t>& indices) {
        return storage->data[index_to_offset(indices)];
    }

    /**
     * @brief Access element via 1D buffer index.
     */
    const T& operator[](size_t idx) const {
        return storage->data[offset + idx];
    }

    /**
     * @brief Access element via 1D buffer index (mutable).
     */
    T& operator[](size_t idx) {
        return storage->data[offset + idx];
    }

    /**
     * @brief Checks if memory layout matches default row-major contiguous strides.
     */
    bool is_contiguous() const {
        return strides == default_strides(shape) && offset == 0;
    }
};

using NDArray = ndarray<float>;

#endif // NDARRAY_HPP
