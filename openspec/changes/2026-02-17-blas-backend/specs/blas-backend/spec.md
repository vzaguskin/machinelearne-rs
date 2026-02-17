## ADDED Requirements

### Requirement: BLAS Backend Implementation

The BlasBackend SHALL implement the Backend trait using optimized BLAS libraries for linear algebra.

#### Scenario: Basic tensor creation

- **GIVEN** the `blas` feature is enabled with a BLAS implementation
- **WHEN** creating tensors via `zeros_1d`, `zeros_2d`, `from_vec_1d`, `from_vec_2d`
- **THEN** tensors are created using ndarray storage
- **AND** data is correctly initialized

#### Scenario: Matrix-vector multiplication

- **WHEN** performing matvec on BLAS tensors
- **THEN** results match CpuBackend within floating-point tolerance
- **AND** BLAS GEMV is used internally

#### Scenario: Matrix-matrix multiplication

- **WHEN** performing matmul on BLAS tensors
- **THEN** results match CpuBackend within floating-point tolerance
- **AND** BLAS GEMM is used internally

### Requirement: Multiple BLAS Implementations

The backend SHALL support multiple BLAS implementations via feature flags.

#### Scenario: OpenBLAS backend

- **GIVEN** the `blas-openblas` feature is enabled
- **WHEN** using BlasBackend
- **THEN** OpenBLAS is linked and used for operations

#### Scenario: Apple Accelerate backend

- **GIVEN** the `blas-accelerate` feature is enabled on macOS
- **WHEN** using BlasBackend
- **THEN** Apple Accelerate framework is used

#### Scenario: Netlib BLAS backend

- **GIVEN** the `blas-netlib` feature is enabled
- **WHEN** using BlasBackend
- **THEN** Netlib BLAS is used

### Requirement: Numerical Correctness

All BLAS operations SHALL produce numerically correct results.

#### Scenario: Matmul accuracy

- **WHEN** multiplying matrices of various sizes
- **THEN** results match CpuBackend within 1e-10 relative error

#### Scenario: Large matrix handling

- **WHEN** multiplying 1000x1000 matrices
- **THEN** operation completes without overflow
- **AND** results are accurate

### Requirement: Performance

BLAS operations SHALL be faster than the pure-Rust CpuBackend.

#### Scenario: Matmul performance

- **WHEN** multiplying matrices larger than 32x32
- **THEN** BlasBackend is significantly faster than CpuBackend
- **AND** speedup increases with matrix size

#### Scenario: Matvec performance

- **WHEN** performing matvec on large matrices (> 100 rows)
- **THEN** BlasBackend is faster than CpuBackend

### Requirement: Error Handling

The backend SHALL handle errors gracefully.

#### Scenario: Shape mismatch

- **WHEN** calling matmul with incompatible shapes
- **THEN** a panic with clear message occurs
- **AND** message indicates the incompatible shapes

#### Scenario: Division by zero

- **WHEN** dividing by a tensor containing zeros
- **THEN** result contains infinities or NaNs as appropriate
- **AND** operation does not panic
