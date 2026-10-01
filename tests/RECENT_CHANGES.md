# Recent Test Changes

This log documents recent test additions, modifications, diffs, and explanations generated during test engineering workflows.

---

## [2026-09-28] Test Suite for `NDArray.relu()`

### 1. Overview & Rationale
- **Target Interface**: `NDArray.relu(self) -> NDArray` in `tests/stubs/GoTorch/backend/native_backend.pyi` and `tests/stubs/GoTorch/backend/ndarray.pyi`.
- **Contract Specification**:
  1. **Element Rectification**: For every element $x$, $\text{ReLU}(x) = \max(0, x)$.
     - Positive numbers ($x > 0$) remain unchanged.
     - Negative numbers ($x < 0$) are clamped to $0.0$.
     - Zeros ($x = 0$, including $-0.0$) remain $0.0$.
  2. **Shape Preservation**: Dimension vector matches input array across arbitrary dimensions (1D, 2D, 3D, 4D).
  3. **Strided & Non-Contiguous Views**: When called on a transposed/strided array (e.g., `a.transpose()`), the operation evaluates logically according to strides and offsets, and returns a contiguous output array.

---

### 2. Diffs & Added Code

#### A. C++ GoogleTest Suite
**File**: [`tests/backend/cppsrc/test_ndarray.cpp`](file:///home/james-paul/AkAutoGrad/tests/backend/cppsrc/test_ndarray.cpp)

```diff
+class ndarray_relu : public ::testing::Test {
+protected:
+    void SetUp() override {}
+    void TearDown() override {}
+};
+
+TEST_F(ndarray_relu, ShouldKeepPositiveValuesUnchanged) {
+    std::vector<float> a_data = {1.0f, 2.5f, 3.0f, 4.5f};
+    std::vector<size_t> shape = {2, 2};
+    NDArray a(a_data, shape);
+
+    NDArray result = a.relu();
+
+    EXPECT_EQ(result.shape, shape);
+    EXPECT_EQ(result.size(), 4);
+    for (size_t i = 0; i < a_data.size(); ++i) {
+        EXPECT_FLOAT_EQ(result[i], a_data[i]);
+    }
+}
+
+TEST_F(ndarray_relu, ShouldZeroOutNegativeValues) {
+    std::vector<float> a_data = {-1.0f, -2.5f, -0.1f, -100.0f};
+    std::vector<size_t> shape = {2, 2};
+    NDArray a(a_data, shape);
+
+    NDArray result = a.relu();
+
+    EXPECT_EQ(result.shape, shape);
+    EXPECT_EQ(result.size(), 4);
+    for (size_t i = 0; i < result.size(); ++i) {
+        EXPECT_FLOAT_EQ(result[i], 0.0f);
+    }
+}
+
+TEST_F(ndarray_relu, ShouldHandleZerosCorrectly) {
+    std::vector<float> a_data = {0.0f, -0.0f, 0.0f};
+    std::vector<size_t> shape = {3};
+    NDArray a(a_data, shape);
+
+    NDArray result = a.relu();
+
+    EXPECT_EQ(result.shape, shape);
+    EXPECT_EQ(result.size(), 3);
+    for (size_t i = 0; i < result.size(); ++i) {
+        EXPECT_FLOAT_EQ(result[i], 0.0f);
+    }
+}
+
+TEST_F(ndarray_relu, ShouldProperlyHandleMixedValues) {
+    std::vector<float> a_data = {-3.0f, 0.0f, 4.5f, -0.5f, 2.0f, -10.0f};
+    std::vector<size_t> shape = {2, 3};
+    NDArray a(a_data, shape);
+
+    NDArray result = a.relu();
+
+    EXPECT_EQ(result.shape, shape);
+    EXPECT_EQ(result.size(), 6);
+    std::vector<float> expected = {0.0f, 0.0f, 4.5f, 0.0f, 2.0f, 0.0f};
+    for (size_t i = 0; i < expected.size(); ++i) {
+        EXPECT_FLOAT_EQ(result[i], expected[i]);
+    }
+}
+
+TEST_F(ndarray_relu, ShouldPreserveMultidimensionalShapes) {
+    std::vector<float> a_data = {
+        -1.0f, 2.0f,
+        0.0f, -3.0f,
+        4.0f, -5.0f,
+        6.0f, -7.0f,
+        -8.0f, 9.0f,
+        10.0f, -11.0f
+    };
+    std::vector<size_t> shape = {2, 3, 2};
+    NDArray a(a_data, shape);
+
+    NDArray result = a.relu();
+
+    EXPECT_EQ(result.shape, shape);
+    EXPECT_EQ(result.size(), 12);
+    std::vector<float> expected = {
+        0.0f, 2.0f,
+        0.0f, 0.0f,
+        4.0f, 0.0f,
+        6.0f, 0.0f,
+        0.0f, 9.0f,
+        10.0f, 0.0f
+    };
+    for (size_t i = 0; i < expected.size(); ++i) {
+        EXPECT_FLOAT_EQ(result[i], expected[i]);
+    }
+}
+
+TEST_F(ndarray_relu, ShouldHandleNonContiguousTransposedStridedArrays) {
+    std::vector<float> a_data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
+    NDArray a(a_data, {2, 3});
+
+    NDArray transposed = a.transpose(0, 1);
+    EXPECT_FALSE(transposed.is_contiguous());
+
+    NDArray result = transposed.relu();
+
+    std::vector<size_t> expected_shape = {3, 2};
+    EXPECT_EQ(result.shape, expected_shape);
+    EXPECT_EQ(result.size(), 6);
+    EXPECT_TRUE(result.is_contiguous());
+
+    std::vector<float> expected_data = {1.0f, 0.0f, 0.0f, 5.0f, 3.0f, 0.0f};
+    for (size_t i = 0; i < expected_data.size(); ++i) {
+        EXPECT_FLOAT_EQ(result[i], expected_data[i]);
+    }
+}
```

#### B. Python Pytest Suite
**File**: [`tests/backend/test_ndarray.py`](file:///home/james-paul/AkAutoGrad/tests/backend/test_ndarray.py)

```diff
+class TestNDArrayRelu:
+    """Verifies rectified linear unit (ReLU) activation behavior."""
+
+    def test_relu_positive_values(self, backend) -> None:
+        """Positive numbers should remain unchanged."""
+        a = backend.NDArray(data=[1.0, 2.5, 3.0, 4.5], shape=(2, 2))
+        res = a.relu()
+        assert res.shape == (2, 2)
+        assert list(res.data) == [1.0, 2.5, 3.0, 4.5]
+
+    def test_relu_negative_values(self, backend) -> None:
+        """Negative numbers should be zeroed out."""
+        a = backend.NDArray(data=[-1.0, -2.5, -0.1, -100.0], shape=(2, 2))
+        res = a.relu()
+        assert res.shape == (2, 2)
+        assert list(res.data) == [0.0, 0.0, 0.0, 0.0]
+
+    def test_relu_zeros(self, backend) -> None:
+        """Zeros should remain zero."""
+        a = backend.NDArray(data=[0.0, 0.0, 0.0], shape=(3,))
+        res = a.relu()
+        assert res.shape == (3,)
+        assert list(res.data) == [0.0, 0.0, 0.0]
+
+    def test_relu_mixed_values_multidimensional(self, backend) -> None:
+        """Mixed positive, negative, and zero values with 3D shape."""
+        data = [
+            -1.0, 2.0,
+            0.0, -3.0,
+            4.0, -5.0,
+            6.0, -7.0,
+            -8.0, 9.0,
+            10.0, -11.0,
+        ]
+        a = backend.NDArray(data=data, shape=(2, 3, 2))
+        res = a.relu()
+        assert res.shape == (2, 3, 2)
+        expected = [
+            0.0, 2.0,
+            0.0, 0.0,
+            4.0, 0.0,
+            6.0, 0.0,
+            0.0, 9.0,
+            10.0, 0.0,
+        ]
+        assert list(res.data) == expected
+
+    def test_relu_strided_non_contiguous(self, backend) -> None:
+        """Tests relu on a non-contiguous transposed view."""
+        a = backend.NDArray(data=[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], shape=(2, 3)).transpose()
+        assert not a.is_contiguous()
+        res = a.relu()
+        assert res.shape == (3, 2)
+        assert res.is_contiguous()
+        assert list(res.data) == [1.0, 0.0, 0.0, 5.0, 3.0, 0.0]
+
+    def test_relu_preserves_shape(self, backend) -> None:
+        """Ensures arbitrary shapes (e.g. 1D, 4D) are preserved after relu."""
+        a_1d = backend.NDArray(data=[-1.0, 1.0], shape=(2,))
+        assert a_1d.relu().shape == (2,)
+
+        shape_4d = (1, 2, 1, 3)
+        data_4d = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]
+        a_4d = backend.NDArray(data=data_4d, shape=shape_4d)
+        res_4d = a_4d.relu()
+        assert res_4d.shape == shape_4d
+        assert list(res_4d.data) == [0.0, 2.0, 0.0, 4.0, 0.0, 6.0]
```

---

### 3. Verification Commands & Status
- **C++ Tests**: `make test-cpp TEST=relu` $\rightarrow$ **6/6 passed** (0 ms)
- **Python Unit Tests**: `PYTHONPATH=. ./.venv/bin/pytest -v tests/backend/test_ndarray.py -k "relu"` $\rightarrow$ **6/6 passed** (0.03 s)
- **Integration Tests**: `PYTHONPATH=. ./.venv/bin/pytest tests/test_tensor.py` $\rightarrow$ **5/5 passed** (0.01 s)

---

## [2026-10-01] Test Suite for `NDArray` Activations, Backward Passes, and `isEmpty()`

### 1. Overview & Rationale
- **Target Interfaces** (from [`tests/stubs/GoTorch/backend/native_backend.pyi`](file:///home/james-paul/AkAutoGrad/tests/stubs/GoTorch/backend/native_backend.pyi)):
  - `isEmpty(self) -> bool`
  - `tanh(self) -> NDArray`
  - `tanh_backward(self, grad_output: NDArray) -> NDArray`
  - `sigmoid(self) -> NDArray`
  - `sigmoid_backward(self, grad_output: NDArray) -> NDArray`
  - `leaky_relu(self, alpha: float = 0.01) -> NDArray`
  - `leaky_relu_backward(self, grad_output: NDArray, alpha: float = 0.01) -> NDArray`
  - `relu_backward(self, grad_output: NDArray) -> NDArray`

- **Contract Specifications**:
  1. **Activation Functions**:
     - `tanh`: Computes $\tanh(x)$, odd symmetry $\tanh(-x) = -\tanh(x)$, saturates at $\pm 1.0$, preserves arbitrary tensor shapes, correctly evaluates strided non-contiguous views.
     - `sigmoid`: Computes $\sigma(x) = \frac{1}{1 + e^{-x}}$, symmetry $\sigma(-x) = 1 - \sigma(x)$, $\sigma(0) = 0.5$, saturates near $0.0$ and $1.0$, preserves arbitrary shapes, evaluates strided non-contiguous views.
     - `leaky_relu`: Computes $x$ for $x > 0$ and $\alpha x$ for $x \le 0$. Default $\alpha = 0.01$, customizable parameter $\alpha$, preserves shapes, evaluates strided non-contiguous views.
  2. **Backward Passes**:
     - `relu_backward`: Upstream gradient passed to elements where $x > 0$, clamped to $0$ where $x \le 0$. Strictly requires `grad_output: NDArray`; raises `TypeError` if omitted or `None`. Raises `ValueError` / `std::invalid_argument` on shape mismatch.
     - `tanh_backward`: Multiplies upstream gradient with $(1 - \tanh^2(x))$. Strictly requires `grad_output: NDArray`.
     - `sigmoid_backward`: Multiplies upstream gradient with $\sigma(x)(1 - \sigma(x))$. Strictly requires `grad_output: NDArray`.
     - `leaky_relu_backward`: Multiplies upstream gradient with $1.0$ ($x > 0$) or $\alpha$ ($x \le 0$). Strictly requires `grad_output: NDArray`.
  3. **Array Inspection**:
     - `isEmpty`: Returns `true` for uninitialized / empty arrays (zero size or empty shape) and `false` for populated arrays.

---

### 2. Diffs & Added Code

#### A. C++ GoogleTest Suite
**File**: [`tests/backend/cppsrc/test_ndarray.cpp`](file:///home/james-paul/AkAutoGrad/tests/backend/cppsrc/test_ndarray.cpp)
- Added `ndarray_relu_backward`:
  - `ShouldComputeGradientForPositiveAndNegativeValues`
  - `ShouldThrowExceptionWhenGradientShapeMismatches`
  - `ShouldHandleNonContiguousTransposedStridedArrays`
- Added `ndarray_tanh`:
  - `ShouldEvaluateTanhCorrectly`
  - `ShouldSatisfyOddSymmetry`
  - `ShouldSaturateForExtremeValues`
  - `ShouldPreserveMultidimensionalShapes`
  - `ShouldHandleNonContiguousTransposedStridedArrays`
- Added `ndarray_tanh_backward`:
  - `ShouldComputeGradientCorrectly`
  - `ShouldThrowExceptionWhenGradientShapeMismatches`
  - `ShouldHandleNonContiguousTransposedStridedArrays`
- Added `ndarray_sigmoid`:
  - `ShouldEvaluateSigmoidCorrectly`
  - `ShouldSatisfySymmetry`
  - `ShouldSaturateForExtremeValues`
  - `ShouldPreserveMultidimensionalShapes`
  - `ShouldHandleNonContiguousTransposedStridedArrays`
- Added `ndarray_sigmoid_backward`:
  - `ShouldComputeGradientCorrectly`
  - `ShouldThrowExceptionWhenGradientShapeMismatches`
  - `ShouldHandleNonContiguousTransposedStridedArrays`
- Added `ndarray_leaky_relu`:
  - `ShouldApplyDefaultAlpha`
  - `ShouldApplyCustomAlpha`
  - `ShouldPreserveMultidimensionalShapes`
  - `ShouldHandleNonContiguousTransposedStridedArrays`
- Added `ndarray_leaky_relu_backward`:
  - `ShouldComputeGradientWithDefaultAlpha`
  - `ShouldComputeGradientWithCustomAlpha`
  - `ShouldThrowExceptionWhenGradientShapeMismatches`
  - `ShouldHandleNonContiguousTransposedStridedArrays`
- Added `ndarray_isEmpty`:
  - `ShouldReturnTrueForDefaultConstructedNDArray`
  - `ShouldReturnFalseForNonEmptyNDArray`

#### B. Python Test Suite
**File**: [`tests/backend/test_ndarray.py`](file:///home/james-paul/AkAutoGrad/tests/backend/test_ndarray.py)
- Added `TestNDArrayReluBackward`:
  - `test_relu_backward_missing_grad_output_raises`
  - `test_relu_backward_none_grad_output_raises`
  - `test_relu_backward_explicit_grad`
  - `test_relu_backward_strided_non_contiguous`
  - `test_relu_backward_shape_mismatch_raises`
- Added `TestNDArrayTanh`:
  - `test_tanh_zero`
  - `test_tanh_values`
  - `test_tanh_odd_symmetry`
  - `test_tanh_saturation`
  - `test_tanh_preserves_shape`
  - `test_tanh_strided_non_contiguous`
- Added `TestNDArrayTanhBackward`:
  - `test_tanh_backward_missing_grad_output_raises`
  - `test_tanh_backward_none_grad_output_raises`
  - `test_tanh_backward_explicit_grad`
  - `test_tanh_backward_strided_non_contiguous`
  - `test_tanh_backward_shape_mismatch_raises`
- Added `TestNDArraySigmoid`:
  - `test_sigmoid_zero`
  - `test_sigmoid_values`
  - `test_sigmoid_symmetry`
  - `test_sigmoid_saturation`
  - `test_sigmoid_preserves_shape`
  - `test_sigmoid_strided_non_contiguous`
- Added `TestNDArraySigmoidBackward`:
  - `test_sigmoid_backward_missing_grad_output_raises`
  - `test_sigmoid_backward_none_grad_output_raises`
  - `test_sigmoid_backward_explicit_grad`
  - `test_sigmoid_backward_strided_non_contiguous`
  - `test_sigmoid_backward_shape_mismatch_raises`
- Added `TestNDArrayLeakyRelu`:
  - `test_leaky_relu_default_alpha`
  - `test_leaky_relu_custom_alpha`
  - `test_leaky_relu_preserves_shape`
  - `test_leaky_relu_strided_non_contiguous`
- Added `TestNDArrayLeakyReluBackward`:
  - `test_leaky_relu_backward_missing_grad_output_raises`
  - `test_leaky_relu_backward_none_grad_output_raises`
  - `test_leaky_relu_backward_default_alpha`
  - `test_leaky_relu_backward_custom_alpha`
  - `test_leaky_relu_backward_strided_non_contiguous`
  - `test_leaky_relu_backward_shape_mismatch_raises`
- Added `TestNDArrayIsEmpty`:
  - `test_is_empty_default_constructed`
  - `test_is_empty_with_data`

---

### 3. Verification Commands & Status
- **C++ GoogleTest Suite**: `make test-cpp` $\rightarrow$ **54/54 passed** (0 ms)
- **Python Backend Unit Tests**: `PYTHONPATH=. ./.venv/bin/pytest -v tests/backend/test_ndarray.py` $\rightarrow$ **52/52 passed** (0.07 s)
- **E2E & Integration Tests**: `PYTHONPATH=. ./.venv/bin/pytest tests/test_tensor.py tests/autograd/test_autograd_engine.py tests/autograd/test_operations.py tests/nn/test_linear.py` $\rightarrow$ **27/27 passed** (1.84 s)

---

## [2026-10-01] Test Suite for 0-D Scalars & Refined `isEmpty()`

### 1. Overview & Rationale
- **Target Contract Updates**:
  1. **0-D Scalars (`num_elements(shape)` with shape `()`)**:
     - Empty shapes `()` / `{}` represent 0-D scalars with `num_elements = 1`.
     - In Python, accessed via `s[()]` or `s.data[0]`. Attempting 1-D indexing `s[0]` raises `IndexError` (dimensional mismatch).
     - Full support for arithmetic operations (e.g. `s1 + s2`) and element-wise activation functions.
  2. **Refined `isEmpty()`**:
     - Verifies whether storage is allocated and contains elements (`!storage || storage->data.empty()`).
     - Returns `true` for uninitialized storage (default constructor).
     - Returns `true` for initialized empty storage (e.g. shape `(0,)` with 0 elements).
     - Returns `false` for initialized populated arrays.
     - Returns `false` for 0-D scalars (`shape = ()`, 1 element).

---

### 2. Diffs & Added Code

#### A. C++ GoogleTest Suite
**File**: [`tests/backend/cppsrc/test_ndarray.cpp`](file:///home/james-paul/AkAutoGrad/tests/backend/cppsrc/test_ndarray.cpp)
- Updated `ndarray_isEmpty`:
  - `ShouldReturnTrueForDefaultConstructedNDArray` (uninitialized storage)
  - `ShouldReturnTrueForInitializedEmptyStorage` (`shape = {0}`, empty data)
  - `ShouldReturnFalseForInitializedNonEmptyStorage` (`shape = {2}`, populated data)
  - `ShouldReturnFalseForScalarNDArray` (`shape = {}`, 1 element)
- Added `ndarray_scalar`:
  - `ShouldRepresentZeroDScalarWithEmptyShapeAndOneElement`
  - `ShouldPerformAdditionBetweenScalars`
  - `ShouldApplyActivationsToScalar`

#### B. Python Test Suite
**File**: [`tests/backend/test_ndarray.py`](file:///home/james-paul/AkAutoGrad/tests/backend/test_ndarray.py)
- Updated `TestNDArrayIsEmpty`:
  - `test_is_empty_default_constructed`
  - `test_is_empty_initialized_empty_storage`
  - `test_is_empty_initialized_non_empty_storage`
  - `test_is_empty_scalar`
- Added `TestNDArrayScalar`:
  - `test_scalar_empty_shape_and_num_elements`: validates shape `()`, size 1, `s[()]`, `s.data[0]`, and `IndexError` on `s[0]`.
  - `test_scalar_addition`: evaluates scalar addition producing scalar with shape `()`.
  - `test_scalar_activations`: evaluates `relu`, `tanh`, `sigmoid`, `leaky_relu`, and backward methods on scalar inputs.

---

### 3. Verification Commands & Status
- **C++ GoogleTest Suite**: `make test-cpp` $\rightarrow$ **59/59 passed** (0 ms)
- **Python Backend Unit Tests**: `PYTHONPATH=. ./.venv/bin/pytest -v tests/backend/test_ndarray.py` $\rightarrow$ **57/57 passed** (0.08 s)
- **E2E & Integration Tests**: `PYTHONPATH=. ./.venv/bin/pytest tests/test_tensor.py tests/autograd/test_autograd_engine.py tests/autograd/test_operations.py tests/nn/test_linear.py` $\rightarrow$ **27/27 passed** (0.99 s)

---

## [2026-10-01] PyTorch Numerical Parity Validation Suite

### 1. Overview & Rationale
- **Target File**: [`tests/backend/test_ndarray_pytorch_validation.py`](file:///home/james-paul/AkAutoGrad/tests/backend/test_ndarray_pytorch_validation.py)
- **Purpose**: Directly validates GoTorch `NDArray` operations against official PyTorch implementations (`torch`) using randomized float inputs.
- **Contract Specification**:
  - Validates that GoTorch `NDArray.tanh()` matches `torch.tanh()` element-by-element within standard float32 tolerance (`rel=1e-5`, `abs=1e-6`) on random uniform floats in range $[-5.0, 5.0]$ across multi-dimensional matrices.

---

### 2. Code Summary
```python
class TestNDArrayPyTorchValidation:
    def test_random_float_tanh_validation(self, backend) -> None:
        random.seed(42)
        shape = (5, 10)
        num_elements = shape[0] * shape[1]
        raw_data = [random.uniform(-5.0, 5.0) for _ in range(num_elements)]

        # GoTorch
        gotorch_array = backend.NDArray(data=raw_data, shape=shape)
        gotorch_result = gotorch_array.tanh()

        # PyTorch
        torch_tensor = torch.tensor(raw_data, dtype=torch.float32).reshape(shape)
        torch_result = torch.tanh(torch_tensor)

        assert gotorch_result.shape == shape
        torch_flattened = torch_result.flatten().tolist()
        for actual, expected in zip(gotorch_result.data, torch_flattened):
            assert pytest.approx(actual, rel=1e-5, abs=1e-6) == expected
```

---

### 3. Verification Commands & Status
- `PYTHONPATH=. ./.venv/bin/pytest -v tests/backend/test_ndarray_pytorch_validation.py` $\rightarrow$ **1/1 passed** (0.78 s)



