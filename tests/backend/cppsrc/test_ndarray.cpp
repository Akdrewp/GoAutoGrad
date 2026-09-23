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
