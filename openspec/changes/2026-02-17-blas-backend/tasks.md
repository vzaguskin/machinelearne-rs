## 1. Dependencies Setup

- [ ] 1.1 Add blas-src dependency to Cargo.toml
- [ ] 1.2 Add feature flags: blas, blas-openblas, blas-netlib, blas-accelerate
- [ ] 1.3 Ensure ndarray dependency is available for blas feature
- [ ] 1.4 Document BLAS library installation requirements

## 2. Backend Implementation

- [ ] 2.1 Create `lib/src/backend/blas.rs` module
- [ ] 2.2 Define BlasTensor2D wrapper around ndarray::Array2
- [ ] 2.3 Implement constructor methods (zeros_1d, zeros_2d, from_vec_*)
- [ ] 2.4 Implement data access methods (to_vec_1d, len_1d, len_2d, shape)
- [ ] 2.5 Implement element-wise operations (delegate to ndarray)
- [ ] 2.6 Implement scalar operations (delegate to ndarray)
- [ ] 2.7 Implement reduction operations (delegate to ndarray)

## 3. BLAS Linear Algebra

- [ ] 3.1 Configure ndarray to use BLAS for dot products
- [ ] 3.2 Implement matvec using BLAS GEMV (via ndarray)
- [ ] 3.3 Implement matvec_transposed using BLAS GEMV
- [ ] 3.4 Implement matmul using BLAS GEMM (via ndarray)
- [ ] 3.5 Implement transpose (use ndarray's optimized transpose)

## 4. Column/Row Operations

- [ ] 4.1 Implement col_mean_2d, col_sum_2d using ndarray
- [ ] 4.2 Implement col_std_2d, col_min_2d, col_max_2d
- [ ] 4.3 Implement row_sum_2d

## 5. Broadcasting Operations

- [ ] 5.1 Implement broadcast_sub_1d_to_2d_rows
- [ ] 5.2 Implement broadcast_div_1d_to_2d_rows
- [ ] 5.3 Implement broadcast_mul_1d_to_2d_rows
- [ ] 5.4 Implement broadcast_add_1d_to_2d_rows

## 6. Other Operations

- [ ] 6.1 Implement mathematical functions (exp, log, sigmoid, sqrt, abs, sign)
- [ ] 6.2 Implement maximum_1d, maximum_2d
- [ ] 6.3 Implement hcat_2d, select_columns_2d, one_hot_from_indices
- [ ] 6.4 Implement ravel_2d

## 7. Testing

- [ ] 7.1 Test all Backend trait methods
- [ ] 7.2 Compare results with CpuBackend for correctness
- [ ] 7.3 Benchmark matmul performance vs CpuBackend
- [ ] 7.4 Benchmark matvec performance vs CpuBackend
- [ ] 7.5 Test with different BLAS implementations

## 8. Documentation

- [ ] 8.1 Add doc comments to BlasBackend and BlasTensor2D
- [ ] 8.2 Document BLAS library installation for each platform
- [ ] 8.3 Update CHANGELOG.md
- [ ] 8.4 Add feature flag documentation
