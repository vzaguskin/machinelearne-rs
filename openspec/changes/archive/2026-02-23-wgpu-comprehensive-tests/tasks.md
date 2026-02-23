## 1. Test Infrastructure

- [x] 1.1 Create `lib/src/backend/wgpu_backend/tests.rs` module
- [x] 1.2 Add test helper functions (comparison utilities, test data generators)
- [x] 1.3 Include tests module in `mod.rs` with `#[cfg(test)]`

## 2. 1D Element-wise Operation Tests

- [x] 2.1 Add tests for `add_1d` across all sizes
- [x] 2.2 Add tests for `sub_1d` across all sizes
- [x] 2.3 Add tests for `mul_1d` across all sizes
- [x] 2.4 Add tests for `div_1d` across all sizes

## 3. 1D Scalar Operation Tests

- [x] 3.1 Add tests for `mul_scalar_1d` across all sizes
- [x] 3.2 Add tests for `add_scalar_1d` across all sizes
- [x] 3.3 Add tests for `sub_scalar_1d` across all sizes
- [x] 3.4 Add tests for `div_scalar_1d` across all sizes

## 4. 2D Element-wise Operation Tests

- [x] 4.1 Add tests for `add_2d` across all sizes
- [x] 4.2 Add tests for `sub_2d` across all sizes
- [x] 4.3 Add tests for `mul_2d` across all sizes
- [x] 4.4 Add tests for `div_2d` across all sizes

## 5. 2D Scalar Operation Tests

- [x] 5.1 Add tests for `mul_scalar_2d` across all sizes
- [x] 5.2 Add tests for `add_scalar_2d` across all sizes
- [x] 5.3 Add tests for `sub_scalar_2d` across all sizes
- [x] 5.4 Add tests for `div_scalar_2d` across all sizes

## 6. Linear Algebra Operation Tests

- [x] 6.1 Add tests for `matvec` across all sizes
- [x] 6.2 Add tests for `matvec_transposed` across all sizes
- [x] 6.3 Add tests for `matmul` across all sizes
- [x] 6.4 Add tests for `transpose` across all sizes

## 7. Reduction Operation Tests

- [x] 7.1 Add tests for `sum_all_1d` across all sizes
- [x] 7.2 Add tests for `sum_all_2d` across all sizes
- [x] 7.3 Add tests for `mean_all_1d` across all sizes
- [x] 7.4 Add tests for `mean_all_2d` across all sizes

## 8. Unary Math Operation Tests

- [x] 8.1 Add tests for `abs_1d` and `abs_2d`
- [x] 8.2 Add tests for `sign_1d` and `sign_2d`
- [x] 8.3 Add tests for `exp_1d` and `exp_2d`
- [x] 8.4 Add tests for `log_1d` and `log_2d`
- [x] 8.5 Add tests for `sigmoid_1d` and `sigmoid_2d`
- [x] 8.6 Add tests for `sqrt_1d` and `sqrt_2d`

## 9. Column/Row Operation Tests

- [x] 9.1 Add tests for `col_mean_2d`
- [x] 9.2 Add tests for `col_std_2d`
- [x] 9.3 Add tests for `col_sum_2d`
- [x] 9.4 Add tests for `col_min_2d` and `col_max_2d`
- [x] 9.5 Add tests for `row_sum_2d`

## 10. Broadcasting Operation Tests

- [x] 10.1 Add tests for `broadcast_add_1d_to_2d_rows`
- [x] 10.2 Add tests for `broadcast_sub_1d_to_2d_rows`
- [x] 10.3 Add tests for `broadcast_mul_1d_to_2d_rows`
- [x] 10.4 Add tests for `broadcast_div_1d_to_2d_rows`

## 11. Tensor Manipulation Operation Tests

- [x] 11.1 Add tests for `ravel_2d`
- [x] 11.2 Add tests for `hcat_2d`
- [x] 11.3 Add tests for `select_columns_2d`
- [x] 11.4 Add tests for `one_hot_from_indices`

## 12. Maximum Operation Tests

- [x] 12.1 Add tests for `maximum_1d`
- [x] 12.2 Add tests for `maximum_2d`

## 13. Integration and Verification

- [x] 13.1 Run all tests and verify they pass
- [x] 13.2 Update CHANGELOG.md
- [x] 13.3 Archive OpenSpec change and create PR
