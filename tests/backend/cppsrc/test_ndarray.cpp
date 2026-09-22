#include <gtest/gtest.h>

#include <stdexcept>
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
