#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ndarray.hpp"

class ndarray_add : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_add, ShouldProperlyAddTwoMatrices) {
    std::vector<float> a_data = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
    std::vector<float> b_data = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    std::vector<size_t> shape = {2, 3};

    NDArray a(a_data, shape);
    NDArray b(b_data, shape);

    NDArray result = a.add(b);

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 6);
    std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_add, ShouldSatisfyCommutativeProperty) {
    NDArray a(std::vector<float>{3.0f, 7.0f, -2.0f, 8.0f}, {2, 2});
    NDArray b(std::vector<float>{5.0f, -1.0f, 4.0f, 2.0f}, {2, 2});

    NDArray ab = a.add(b);
    NDArray ba = b.add(a);

    EXPECT_EQ(ab.shape, ba.shape);
    for (size_t i = 0; i < ab.size(); ++i) {
        EXPECT_FLOAT_EQ(ab[i], ba[i]);
    }
}

TEST_F(ndarray_add, ShouldBroadcastWhenOneTensorHasMissingLeadingDimension) {
    // A: shape (2, 3)
    // B: shape (3,) -> padded on left to (1, 3) -> broadcast to (2, 3)
    NDArray a(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    NDArray b(std::vector<float>{10.0f, 20.0f, 30.0f}, {3});

    NDArray result = a.add(b);

    std::vector<size_t> expected_shape = {2, 3};
    EXPECT_EQ(result.shape, expected_shape);

    std::vector<float> expected_data = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

TEST_F(ndarray_add, ShouldBroadcastWhenDimensionIsOne) {
    // A: shape (2, 1)
    // B: shape (1, 3)
    // Result: shape (2, 3)
    NDArray a(std::vector<float>{1.0f, 2.0f}, {2, 1});
    NDArray b(std::vector<float>{10.0f, 20.0f, 30.0f}, {1, 3});

    NDArray result = a.add(b);

    std::vector<size_t> expected_shape = {2, 3};
    EXPECT_EQ(result.shape, expected_shape);

    // Row 0: 1 + [10, 20, 30] = [11, 21, 31]
    // Row 1: 2 + [10, 20, 30] = [12, 22, 32]
    std::vector<float> expected_data = {11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f};
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

TEST_F(ndarray_add, ShouldThrowExceptionWhenMiddleDimensionMismatches) {
    // A: shape (2, 3, 4)
    // B: shape (2, 5, 4) -> middle dimension 3 vs 5, neither is 1 -> mismatch!
    NDArray a(std::vector<float>(24, 1.0f), {2, 3, 4});
    NDArray b(std::vector<float>(40, 2.0f), {2, 5, 4});

    EXPECT_THROW(a.add(b), std::invalid_argument);
    EXPECT_THROW(b.add(a), std::invalid_argument);
}

TEST_F(ndarray_add, ShouldBroadcastFrontDimension) {
    // A: shape (1, 3, 2) -> front dimension 1
    // B: shape (2, 3, 2)
    // Result: shape (2, 3, 2)
    NDArray a(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1, 3, 2});
    NDArray b(std::vector<float>{
        10.0f, 10.0f, 20.0f, 20.0f, 30.0f, 30.0f,
        40.0f, 40.0f, 50.0f, 50.0f, 60.0f, 60.0f
    }, {2, 3, 2});

    NDArray result = a.add(b);

    std::vector<size_t> expected_shape = {2, 3, 2};
    EXPECT_EQ(result.shape, expected_shape);

    std::vector<float> expected_data = {
        11.0f, 12.0f, 23.0f, 24.0f, 35.0f, 36.0f,
        41.0f, 42.0f, 53.0f, 54.0f, 65.0f, 66.0f
    };
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

TEST_F(ndarray_add, ShouldBroadcastBackDimension) {
    // A: shape (2, 3, 1) -> back dimension 1
    // B: shape (2, 3, 2)
    // Result: shape (2, 3, 2)
    NDArray a(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3, 1});
    NDArray b(std::vector<float>{
        10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f,
        70.0f, 80.0f, 90.0f, 100.0f, 110.0f, 120.0f
    }, {2, 3, 2});

    NDArray result = a.add(b);

    std::vector<size_t> expected_shape = {2, 3, 2};
    EXPECT_EQ(result.shape, expected_shape);

    std::vector<float> expected_data = {
        11.0f, 21.0f, 32.0f, 42.0f, 53.0f, 63.0f,
        74.0f, 84.0f, 95.0f, 105.0f, 116.0f, 126.0f
    };
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

TEST_F(ndarray_add, ShouldBroadcastMiddleDimension) {
    // A: shape (2, 1, 2) -> middle dimension 1
    // B: shape (2, 3, 2)
    // Result: shape (2, 3, 2)
    NDArray a(std::vector<float>{1.0f, 2.0f, 10.0f, 20.0f}, {2, 1, 2});
    NDArray b(std::vector<float>{
        1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f,
        4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f
    }, {2, 3, 2});

    NDArray result = a.add(b);

    std::vector<size_t> expected_shape = {2, 3, 2};
    EXPECT_EQ(result.shape, expected_shape);

    std::vector<float> expected_data = {
        2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f,
        14.0f, 24.0f, 15.0f, 25.0f, 16.0f, 26.0f
    };
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

class ndarray_mul : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_mul, ShouldProperlyMultiplyMatricesWithNormalNumbers) {
    // A: shape (2, 3), B: shape (3, 2) -> Result: shape (2, 2)
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
    NDArray a(a_data, {2, 3});
    NDArray b(b_data, {3, 2});

    NDArray result = a.matmul(b);

    std::vector<size_t> expected_shape = {2, 2};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_EQ(result.size(), 4);

    // Row 0: [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [31, 19]
    // Row 1: [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [85, 55]
    std::vector<float> expected_data = {31.0f, 19.0f, 85.0f, 55.0f};
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

TEST_F(ndarray_mul, ShouldMultiplyWhenOneMatrixIsZero) {
    NDArray a(std::vector<float>{1.5f, -2.0f, 3.0f, 4.2f}, {2, 2});
    NDArray zero(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, {2, 2});

    // A * 0 = 0
    NDArray result1 = a.matmul(zero);
    EXPECT_EQ(result1.shape, (std::vector<size_t>{2, 2}));
    for (size_t i = 0; i < result1.size(); ++i) {
        EXPECT_FLOAT_EQ(result1[i], 0.0f);
    }

    // 0 * A = 0
    NDArray result2 = zero.matmul(a);
    EXPECT_EQ(result2.shape, (std::vector<size_t>{2, 2}));
    for (size_t i = 0; i < result2.size(); ++i) {
        EXPECT_FLOAT_EQ(result2[i], 0.0f);
    }
}

TEST_F(ndarray_mul, ShouldMultiplyWhenOneMatrixIsIdentity) {
    std::vector<float> a_data = {3.0f, -1.0f, 4.0f, 2.5f};
    NDArray a(a_data, {2, 2});
    NDArray identity(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}, {2, 2});

    // A * I = A
    NDArray result1 = a.matmul(identity);
    EXPECT_EQ(result1.shape, a.shape);
    for (size_t i = 0; i < a_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result1[i], a_data[i]);
    }

    // I * A = A
    NDArray result2 = identity.matmul(a);
    EXPECT_EQ(result2.shape, a.shape);
    for (size_t i = 0; i < a_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result2[i], a_data[i]);
    }
}

TEST_F(ndarray_mul, ShouldThrowExceptionWhenDimensionsDoNotMatch) {
    // Inner dimensions mismatch: (2, 3) @ (4, 2) -> 3 != 4
    NDArray a(std::vector<float>(6, 1.0f), {2, 3});
    NDArray b(std::vector<float>(8, 1.0f), {4, 2});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);

    // Non-2D operands: 1D or 3D
    NDArray c_1d(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.matmul(c_1d), std::invalid_argument);

    NDArray d_3d(std::vector<float>(24, 1.0f), {2, 3, 4});
    EXPECT_THROW(d_3d.matmul(a), std::invalid_argument);
}

TEST_F(ndarray_mul, ShouldFailWithIncompatibleTypes) {
    // In C++, strong typing prevents cross-type matrix multiplication between
    // ndarray<float> and incompatible types (e.g. ndarray<int> or ndarray<double>).
    EXPECT_FALSE((std::is_invocable_v<decltype(&ndarray<float>::matmul), const ndarray<float>*, const ndarray<int>&>));
    EXPECT_FALSE((std::is_invocable_v<decltype(&ndarray<float>::matmul), const ndarray<float>*, const ndarray<double>&>));
}

TEST_F(ndarray_mul, ShouldHandleTooLargeNumbersForDataType) {
    // Multiplying numbers that exceed float dynamic range results in infinity (IEEE 754 overflow)
    float large_val = 1e30f;
    NDArray a(std::vector<float>{large_val, large_val, large_val, large_val}, {2, 2});
    NDArray b(std::vector<float>{large_val, large_val, large_val, large_val}, {2, 2});

    NDArray result = a.matmul(b);

    EXPECT_EQ(result.shape, (std::vector<size_t>{2, 2}));
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isinf(result[i]));
    }
}

class ndarray_sub : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_sub, ShouldProperlySubtractTwoMatrices) {
    std::vector<float> a_data = {5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f};
    std::vector<float> b_data = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
    std::vector<size_t> shape = {2, 3};

    NDArray a(a_data, shape);
    NDArray b(b_data, shape);

    NDArray result = a.sub(b);

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 6);
    std::vector<float> expected = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }

    // Verify operator-
    NDArray op_result = a - b;
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(op_result[i], expected[i]);
    }
}

TEST_F(ndarray_sub, ShouldNotBeCommutative) {
    NDArray a(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f}, {2, 2});
    NDArray b(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    NDArray ab = a.sub(b);
    NDArray ba = b.sub(a);

    EXPECT_EQ(ab.shape, ba.shape);
    for (size_t i = 0; i < ab.size(); ++i) {
        EXPECT_FLOAT_EQ(ab[i], -ba[i]);
    }
}

TEST_F(ndarray_sub, ShouldBroadcastWhenOneTensorHasMissingLeadingDimension) {
    // A: shape (2, 3), B: shape (3,) -> padded to (1, 3) -> broadcast to (2, 3)
    NDArray a(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3});
    NDArray b(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});

    NDArray result = a.sub(b);

    std::vector<size_t> expected_shape = {2, 3};
    EXPECT_EQ(result.shape, expected_shape);

    std::vector<float> expected_data = {9.0f, 18.0f, 27.0f, 39.0f, 48.0f, 57.0f};
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

TEST_F(ndarray_sub, ShouldBroadcastWhenDimensionIsOne) {
    // A: shape (2, 1), B: shape (1, 3) -> Result: shape (2, 3)
    NDArray a(std::vector<float>{10.0f, 20.0f}, {2, 1});
    NDArray b(std::vector<float>{1.0f, 2.0f, 3.0f}, {1, 3});

    NDArray result = a.sub(b);

    std::vector<size_t> expected_shape = {2, 3};
    EXPECT_EQ(result.shape, expected_shape);

    // Row 0: 10 - [1, 2, 3] = [9, 8, 7]
    // Row 1: 20 - [1, 2, 3] = [19, 18, 17]
    std::vector<float> expected_data = {9.0f, 8.0f, 7.0f, 19.0f, 18.0f, 17.0f};
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

TEST_F(ndarray_sub, ShouldThrowExceptionWhenMiddleDimensionMismatches) {
    NDArray a(std::vector<float>(24, 1.0f), {2, 3, 4});
    NDArray b(std::vector<float>(40, 2.0f), {2, 5, 4});

    EXPECT_THROW(a.sub(b), std::invalid_argument);
    EXPECT_THROW(b.sub(a), std::invalid_argument);
}

class ndarray_relu : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_relu, ShouldKeepPositiveValuesUnchanged) {
    std::vector<float> a_data = {1.0f, 2.5f, 3.0f, 4.5f};
    std::vector<size_t> shape = {2, 2};
    NDArray a(a_data, shape);

    NDArray result = a.relu();

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 4);
    for (size_t i = 0; i < a_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], a_data[i]);
    }
}

TEST_F(ndarray_relu, ShouldZeroOutNegativeValues) {
    std::vector<float> a_data = {-1.0f, -2.5f, -0.1f, -100.0f};
    std::vector<size_t> shape = {2, 2};
    NDArray a(a_data, shape);

    NDArray result = a.relu();

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 4);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

TEST_F(ndarray_relu, ShouldHandleZerosCorrectly) {
    std::vector<float> a_data = {0.0f, -0.0f, 0.0f};
    std::vector<size_t> shape = {3};
    NDArray a(a_data, shape);

    NDArray result = a.relu();

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 3);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

TEST_F(ndarray_relu, ShouldProperlyHandleMixedValues) {
    std::vector<float> a_data = {-3.0f, 0.0f, 4.5f, -0.5f, 2.0f, -10.0f};
    std::vector<size_t> shape = {2, 3};
    NDArray a(a_data, shape);

    NDArray result = a.relu();

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 6);
    std::vector<float> expected = {0.0f, 0.0f, 4.5f, 0.0f, 2.0f, 0.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_relu, ShouldPreserveMultidimensionalShapes) {
    std::vector<float> a_data = {
        -1.0f, 2.0f,
        0.0f, -3.0f,
        4.0f, -5.0f,
        6.0f, -7.0f,
        -8.0f, 9.0f,
        10.0f, -11.0f
    };
    std::vector<size_t> shape = {2, 3, 2};
    NDArray a(a_data, shape);

    NDArray result = a.relu();

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 12);
    std::vector<float> expected = {
        0.0f, 2.0f,
        0.0f, 0.0f,
        4.0f, 0.0f,
        6.0f, 0.0f,
        0.0f, 9.0f,
        10.0f, 0.0f
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_relu, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    NDArray a(a_data, {2, 3});

    NDArray transposed = a.transpose(0, 1);
    EXPECT_FALSE(transposed.is_contiguous());

    NDArray result = transposed.relu();

    std::vector<size_t> expected_shape = {3, 2};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_EQ(result.size(), 6);
    EXPECT_TRUE(result.is_contiguous());

    std::vector<float> expected_data = {1.0f, 0.0f, 0.0f, 5.0f, 3.0f, 0.0f};
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
    }
}

class ndarray_relu_backward : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_relu_backward, ShouldComputeGradientForPositiveAndNegativeValues) {
    std::vector<float> a_data = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f, 5.0f};
    std::vector<float> grad_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<size_t> shape = {2, 3};
    NDArray a(a_data, shape);
    NDArray grad(grad_data, shape);

    NDArray result = a.relu_backward(grad);

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 6);
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_relu_backward, ShouldThrowExceptionWhenGradientShapeMismatches) {
    NDArray a(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    NDArray grad(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});

    EXPECT_THROW(a.relu_backward(grad), std::invalid_argument);
}

TEST_F(ndarray_relu_backward, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    NDArray a(a_data, {2, 3});
    NDArray a_t = a.transpose(0, 1);
    EXPECT_FALSE(a_t.is_contiguous());

    std::vector<float> grad_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    NDArray grad(grad_data, {3, 2});

    NDArray result = a_t.relu_backward(grad);

    std::vector<size_t> expected_shape = {3, 2};
    EXPECT_EQ(result.shape, expected_shape);
    EXPECT_TRUE(result.is_contiguous());
    // a_t elements: [1.0, -4.0, -2.0, 5.0, 3.0, -6.0]
    // grad elements: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    std::vector<float> expected = {10.0f, 0.0f, 0.0f, 40.0f, 50.0f, 0.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

class ndarray_tanh : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_tanh, ShouldEvaluateTanhCorrectly) {
    std::vector<float> a_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<size_t> shape = {5};
    NDArray a(a_data, shape);

    NDArray result = a.tanh();

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 5);
    for (size_t i = 0; i < a_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], std::tanh(a_data[i]));
    }
}

TEST_F(ndarray_tanh, ShouldSatisfyOddSymmetry) {
    NDArray a(std::vector<float>{-3.0f, -1.5f, 0.5f, 2.0f}, {2, 2});
    NDArray neg_a(std::vector<float>{3.0f, 1.5f, -0.5f, -2.0f}, {2, 2});

    NDArray res = a.tanh();
    NDArray res_neg = neg_a.tanh();

    for (size_t i = 0; i < res.size(); ++i) {
        EXPECT_FLOAT_EQ(res[i], -res_neg[i]);
    }
}

TEST_F(ndarray_tanh, ShouldSaturateForExtremeValues) {
    NDArray a(std::vector<float>{-40.0f, 40.0f}, {2});
    NDArray result = a.tanh();

    EXPECT_FLOAT_EQ(result[0], -1.0f);
    EXPECT_FLOAT_EQ(result[1], 1.0f);
}

TEST_F(ndarray_tanh, ShouldPreserveMultidimensionalShapes) {
    std::vector<float> a_data(12, 0.5f);
    std::vector<size_t> shape = {2, 3, 2};
    NDArray a(a_data, shape);

    NDArray result = a.tanh();
    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 12);
}

TEST_F(ndarray_tanh, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 0.5f};
    NDArray a(a_data, {2, 3});
    NDArray a_t = a.transpose(0, 1);
    EXPECT_FALSE(a_t.is_contiguous());

    NDArray result = a_t.tanh();
    EXPECT_EQ(result.shape, (std::vector<size_t>{3, 2}));
    EXPECT_TRUE(result.is_contiguous());

    // a_t elements: [0.0, -1.0, 1.0, -2.0, 2.0, 0.5]
    std::vector<float> expected = {
        std::tanh(0.0f), std::tanh(-1.0f),
        std::tanh(1.0f), std::tanh(-2.0f),
        std::tanh(2.0f), std::tanh(0.5f)
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

class ndarray_tanh_backward : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_tanh_backward, ShouldComputeGradientCorrectly) {
    std::vector<float> a_data = {-2.0f, 0.0f, 1.5f};
    std::vector<float> grad_data = {2.0f, 3.0f, 4.0f};
    std::vector<size_t> shape = {3};
    NDArray a(a_data, shape);
    NDArray grad(grad_data, shape);

    NDArray result = a.tanh_backward(grad);

    EXPECT_EQ(result.shape, shape);
    // d/dx tanh(x) = (1 - tanh^2(x)) * grad
    for (size_t i = 0; i < a_data.size(); ++i) {
        float t = std::tanh(a_data[i]);
        float expected = (1.0f - t * t) * grad_data[i];
        EXPECT_FLOAT_EQ(result[i], expected);
    }
}

TEST_F(ndarray_tanh_backward, ShouldThrowExceptionWhenGradientShapeMismatches) {
    NDArray a(std::vector<float>{1.0f, 2.0f}, {2});
    NDArray grad(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});

    EXPECT_THROW(a.tanh_backward(grad), std::invalid_argument);
}

TEST_F(ndarray_tanh_backward, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {0.0f, 1.0f, -1.0f, 2.0f};
    NDArray a(a_data, {2, 2});
    NDArray a_t = a.transpose(0, 1);

    NDArray grad(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    NDArray result = a_t.tanh_backward(grad);

    EXPECT_EQ(result.shape, (std::vector<size_t>{2, 2}));
    EXPECT_TRUE(result.is_contiguous());

    // a_t: [0.0, -1.0, 1.0, 2.0]
    std::vector<float> at_vals = {0.0f, -1.0f, 1.0f, 2.0f};
    std::vector<float> g_vals = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < 4; ++i) {
        float t = std::tanh(at_vals[i]);
        EXPECT_FLOAT_EQ(result[i], (1.0f - t * t) * g_vals[i]);
    }
}

class ndarray_sigmoid : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_sigmoid, ShouldEvaluateSigmoidCorrectly) {
    std::vector<float> a_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<size_t> shape = {5};
    NDArray a(a_data, shape);

    NDArray result = a.sigmoid();

    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 5);
    for (size_t i = 0; i < a_data.size(); ++i) {
        float s = 1.0f / (1.0f + std::exp(-a_data[i]));
        EXPECT_FLOAT_EQ(result[i], s);
    }
}

TEST_F(ndarray_sigmoid, ShouldSatisfySymmetry) {
    NDArray a(std::vector<float>{-3.0f, -1.5f, 0.5f, 2.0f}, {2, 2});
    NDArray neg_a(std::vector<float>{3.0f, 1.5f, -0.5f, -2.0f}, {2, 2});

    NDArray res = a.sigmoid();
    NDArray res_neg = neg_a.sigmoid();

    // sigma(-x) == 1 - sigma(x)
    for (size_t i = 0; i < res.size(); ++i) {
        EXPECT_NEAR(res[i] + res_neg[i], 1.0f, 1e-6f);
    }
}

TEST_F(ndarray_sigmoid, ShouldSaturateForExtremeValues) {
    NDArray a(std::vector<float>{-50.0f, 50.0f}, {2});
    NDArray result = a.sigmoid();

    EXPECT_NEAR(result[0], 0.0f, 1e-6f);
    EXPECT_FLOAT_EQ(result[1], 1.0f);
}

TEST_F(ndarray_sigmoid, ShouldPreserveMultidimensionalShapes) {
    std::vector<float> a_data(8, 0.1f);
    std::vector<size_t> shape = {2, 2, 2};
    NDArray a(a_data, shape);

    NDArray result = a.sigmoid();
    EXPECT_EQ(result.shape, shape);
    EXPECT_EQ(result.size(), 8);
}

TEST_F(ndarray_sigmoid, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 0.5f};
    NDArray a(a_data, {2, 3});
    NDArray a_t = a.transpose(0, 1);
    EXPECT_FALSE(a_t.is_contiguous());

    NDArray result = a_t.sigmoid();
    EXPECT_EQ(result.shape, (std::vector<size_t>{3, 2}));
    EXPECT_TRUE(result.is_contiguous());

    std::vector<float> expected = {
        1.0f / (1.0f + std::exp(-0.0f)),
        1.0f / (1.0f + std::exp(1.0f)),
        1.0f / (1.0f + std::exp(-1.0f)),
        1.0f / (1.0f + std::exp(2.0f)),
        1.0f / (1.0f + std::exp(-2.0f)),
        1.0f / (1.0f + std::exp(-0.5f))
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

class ndarray_sigmoid_backward : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_sigmoid_backward, ShouldComputeGradientCorrectly) {
    std::vector<float> a_data = {-2.0f, 0.0f, 2.0f};
    std::vector<float> grad_data = {1.5f, 2.5f, 3.5f};
    std::vector<size_t> shape = {3};
    NDArray a(a_data, shape);
    NDArray grad(grad_data, shape);

    NDArray result = a.sigmoid_backward(grad);

    EXPECT_EQ(result.shape, shape);
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) * grad
    for (size_t i = 0; i < a_data.size(); ++i) {
        float s = 1.0f / (1.0f + std::exp(-a_data[i]));
        float expected = s * (1.0f - s) * grad_data[i];
        EXPECT_FLOAT_EQ(result[i], expected);
    }
}

TEST_F(ndarray_sigmoid_backward, ShouldThrowExceptionWhenGradientShapeMismatches) {
    NDArray a(std::vector<float>{1.0f, 2.0f}, {2});
    NDArray grad(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});

    EXPECT_THROW(a.sigmoid_backward(grad), std::invalid_argument);
}

TEST_F(ndarray_sigmoid_backward, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {0.0f, 1.0f, -1.0f, 2.0f};
    NDArray a(a_data, {2, 2});
    NDArray a_t = a.transpose(0, 1);

    NDArray grad(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    NDArray result = a_t.sigmoid_backward(grad);

    EXPECT_EQ(result.shape, (std::vector<size_t>{2, 2}));
    EXPECT_TRUE(result.is_contiguous());

    std::vector<float> at_vals = {0.0f, -1.0f, 1.0f, 2.0f};
    std::vector<float> g_vals = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < 4; ++i) {
        float s = 1.0f / (1.0f + std::exp(-at_vals[i]));
        EXPECT_FLOAT_EQ(result[i], s * (1.0f - s) * g_vals[i]);
    }
}

class ndarray_leaky_relu : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_leaky_relu, ShouldApplyDefaultAlpha) {
    std::vector<float> a_data = {-200.0f, -10.0f, 0.0f, 5.0f, 10.0f};
    std::vector<size_t> shape = {5};
    NDArray a(a_data, shape);

    NDArray result = a.leaky_relu();

    EXPECT_EQ(result.shape, shape);
    std::vector<float> expected = {-2.0f, -0.1f, 0.0f, 5.0f, 10.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_leaky_relu, ShouldApplyCustomAlpha) {
    std::vector<float> a_data = {-10.0f, -5.0f, 0.0f, 4.0f};
    std::vector<size_t> shape = {2, 2};
    NDArray a(a_data, shape);

    float alpha = 0.2f;
    NDArray result = a.leaky_relu(alpha);

    EXPECT_EQ(result.shape, shape);
    std::vector<float> expected = {-2.0f, -1.0f, 0.0f, 4.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_leaky_relu, ShouldPreserveMultidimensionalShapes) {
    std::vector<float> a_data(12, -2.0f);
    std::vector<size_t> shape = {2, 3, 2};
    NDArray a(a_data, shape);

    NDArray result = a.leaky_relu(0.1f);
    EXPECT_EQ(result.shape, shape);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(result[i], -0.2f);
    }
}

TEST_F(ndarray_leaky_relu, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {10.0f, -20.0f, 30.0f, -40.0f, 50.0f, -60.0f};
    NDArray a(a_data, {2, 3});
    NDArray a_t = a.transpose(0, 1);
    EXPECT_FALSE(a_t.is_contiguous());

    NDArray result = a_t.leaky_relu(0.1f);
    EXPECT_EQ(result.shape, (std::vector<size_t>{3, 2}));
    EXPECT_TRUE(result.is_contiguous());

    // a_t elements: [10.0, -40.0, -20.0, 50.0, 30.0, -60.0]
    std::vector<float> expected = {10.0f, -4.0f, -2.0f, 50.0f, 30.0f, -6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

class ndarray_leaky_relu_backward : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_leaky_relu_backward, ShouldComputeGradientWithDefaultAlpha) {
    std::vector<float> a_data = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f};
    std::vector<float> grad_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::vector<size_t> shape = {5};
    NDArray a(a_data, shape);
    NDArray grad(grad_data, shape);

    NDArray result = a.leaky_relu_backward(grad);

    EXPECT_EQ(result.shape, shape);
    // x > 0: grad, x <= 0: 0.01 * grad
    std::vector<float> expected = {0.1f, 0.2f, 0.3f, 40.0f, 50.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_leaky_relu_backward, ShouldComputeGradientWithCustomAlpha) {
    std::vector<float> a_data = {-2.0f, 0.0f, 3.0f};
    std::vector<float> grad_data = {10.0f, 20.0f, 30.0f};
    std::vector<size_t> shape = {3};
    NDArray a(a_data, shape);
    NDArray grad(grad_data, shape);

    float alpha = 0.25f;
    NDArray result = a.leaky_relu_backward(grad, alpha);

    EXPECT_EQ(result.shape, shape);
    std::vector<float> expected = {2.5f, 5.0f, 30.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

TEST_F(ndarray_leaky_relu_backward, ShouldThrowExceptionWhenGradientShapeMismatches) {
    NDArray a(std::vector<float>{1.0f, 2.0f}, {2});
    NDArray grad(std::vector<float>{1.0f, 2.0f, 3.0f}, {3});

    EXPECT_THROW(a.leaky_relu_backward(grad), std::invalid_argument);
}

TEST_F(ndarray_leaky_relu_backward, ShouldHandleNonContiguousTransposedStridedArrays) {
    std::vector<float> a_data = {1.0f, -2.0f, 3.0f, -4.0f};
    NDArray a(a_data, {2, 2});
    NDArray a_t = a.transpose(0, 1);

    NDArray grad(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f}, {2, 2});
    NDArray result = a_t.leaky_relu_backward(grad, 0.1f);

    EXPECT_EQ(result.shape, (std::vector<size_t>{2, 2}));
    EXPECT_TRUE(result.is_contiguous());

    // a_t: [1.0, 3.0, -2.0, -4.0]
    // grad: [10.0, 20.0, 30.0, 40.0]
    std::vector<float> expected = {10.0f, 20.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]);
    }
}

class ndarray_isEmpty : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_isEmpty, ShouldReturnTrueForDefaultConstructedNDArray) {
    // Uninitialized storage
    NDArray a;
    EXPECT_TRUE(a.isEmpty());
}

TEST_F(ndarray_isEmpty, ShouldReturnTrueForInitializedEmptyStorage) {
    // Initialized empty storage (0 total elements)
    NDArray a(std::vector<float>{}, {0});
    EXPECT_TRUE(a.isEmpty());
}

TEST_F(ndarray_isEmpty, ShouldReturnFalseForInitializedNonEmptyStorage) {
    // Initialized non-empty storage
    NDArray a(std::vector<float>{1.0f, 2.0f}, {2});
    EXPECT_FALSE(a.isEmpty());
}

TEST_F(ndarray_isEmpty, ShouldReturnFalseForScalarNDArray) {
    // 0-D scalar has 1 element, hence not empty
    NDArray a(std::vector<float>{42.0f}, {});
    EXPECT_FALSE(a.isEmpty());
}

class ndarray_scalar : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ndarray_scalar, ShouldRepresentZeroDScalarWithEmptyShapeAndOneElement) {
    NDArray s(std::vector<float>{3.14f}, {});
    EXPECT_EQ(s.shape, (std::vector<size_t>{}));
    EXPECT_EQ(s.size(), 1);
    EXPECT_FLOAT_EQ(s[0], 3.14f);
    EXPECT_FALSE(s.isEmpty());
}

TEST_F(ndarray_scalar, ShouldPerformAdditionBetweenScalars) {
    NDArray s1(std::vector<float>{2.5f}, {});
    NDArray s2(std::vector<float>{3.5f}, {});
    NDArray res = s1.add(s2);
    EXPECT_EQ(res.shape, (std::vector<size_t>{}));
    EXPECT_EQ(res.size(), 1);
    EXPECT_FLOAT_EQ(res[0], 6.0f);
}

TEST_F(ndarray_scalar, ShouldApplyActivationsToScalar) {
    NDArray s(std::vector<float>{-2.0f}, {});
    EXPECT_FLOAT_EQ(s.relu()[0], 0.0f);
    EXPECT_FLOAT_EQ(s.tanh()[0], std::tanh(-2.0f));
    EXPECT_FLOAT_EQ(s.sigmoid()[0], 1.0f / (1.0f + std::exp(2.0f)));
    EXPECT_FLOAT_EQ(s.leaky_relu(0.1f)[0], -0.2f);

    NDArray grad(std::vector<float>{5.0f}, {});
    EXPECT_FLOAT_EQ(s.relu_backward(grad)[0], 0.0f);
    EXPECT_FLOAT_EQ(s.leaky_relu_backward(grad, 0.1f)[0], 0.5f);
}




