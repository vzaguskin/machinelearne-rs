## ADDED Requirements

### Requirement: WGPU Backend Implementation

The WgpuBackend SHALL implement the Backend trait with GPU-accelerated operations.

#### Scenario: Basic tensor creation

- **GIVEN** the `wgpu` feature is enabled
- **WHEN** creating tensors via `zeros_1d`, `zeros_2d`, `from_vec_1d`, `from_vec_2d`
- **THEN** tensors are created on the GPU device
- **AND** data is correctly transferred to GPU memory

#### Scenario: Element-wise operations

- **WHEN** performing element-wise operations (add, sub, mul, div) on GPU tensors
- **THEN** results match CpuBackend within floating-point tolerance
- **AND** operations execute on GPU

#### Scenario: Linear algebra operations

- **WHEN** performing matvec or matmul on GPU tensors
- **THEN** results match CpuBackend within floating-point tolerance
- **AND** operations are faster than CpuBackend for sufficiently large tensors

### Requirement: Device Management

The WgpuBackend SHALL support device selection and management.

#### Scenario: Default device selection

- **WHEN** using WgpuBackend without specifying a device
- **THEN** a suitable GPU device is automatically selected
- **AND** if no GPU is available, returns a clear error

#### Scenario: Device enumeration

- **WHEN** enumerating available devices
- **THEN** all available GPU adapters are listed
- **AND** device names and capabilities are available

### Requirement: Numerical Correctness

All GPU operations SHALL produce numerically correct results.

#### Scenario: Sigmoid stability

- **WHEN** computing sigmoid of large positive or negative values
- **THEN** results are within [0, 1] without overflow
- **AND** results match the numerically stable CPU implementation

#### Scenario: Matrix multiplication

- **WHEN** multiplying matrices of various sizes
- **THEN** results match CPU matmul within 1e-5 relative error

### Requirement: Error Handling

The backend SHALL handle GPU errors gracefully.

#### Scenario: Out of memory

- **WHEN** allocating a tensor that exceeds GPU memory
- **THEN** a clear error is returned (not panic)
- **AND** error message indicates memory constraint

#### Scenario: Device lost

- **WHEN** GPU device is lost during operation
- **THEN** operation fails with appropriate error
- **AND** subsequent operations can reinitialize

### Requirement: Performance

GPU operations SHALL provide speedup over CPU for appropriate tensor sizes.

#### Scenario: Matrix-vector multiplication performance

- **WHEN** performing matvec on tensors larger than threshold
- **THEN** GPU execution is faster than CpuBackend
- **AND** threshold is documented (estimated: matrices > 64x64)
