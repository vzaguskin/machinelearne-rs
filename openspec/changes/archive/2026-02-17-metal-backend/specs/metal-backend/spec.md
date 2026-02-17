## ADDED Requirements

### Requirement: Metal Backend Implementation

The MetalBackend SHALL implement the Backend trait with Apple GPU acceleration.

#### Scenario: Basic tensor creation

- **GIVEN** the `metal` feature is enabled on macOS
- **WHEN** creating tensors via `zeros_1d`, `zeros_2d`, `from_vec_1d`, `from_vec_2d`
- **THEN** tensors are allocated in shared GPU memory
- **AND** CPU access is zero-copy

#### Scenario: Matrix multiplication with MPS

- **WHEN** performing matmul on Metal tensors
- **THEN** Metal Performance Shaders are used
- **AND** results match CpuBackend within floating-point tolerance

#### Scenario: Custom shader operations

- **WHEN** performing operations not in MPS (e.g., sigmoid)
- **THEN** custom Metal shaders are executed
- **AND** results match CpuBackend within floating-point tolerance

### Requirement: Unified Memory

The backend SHALL leverage unified memory on Apple Silicon.

#### Scenario: Zero-copy CPU access

- **WHEN** creating a tensor from CPU data
- **THEN** no copy is made for the data transfer
- **AND** CPU can read GPU-computed results directly

#### Scenario: Memory efficiency

- **WHEN** allocating large tensors
- **THEN** only one copy exists in unified memory
- **AND** no staging buffers are needed

### Requirement: Device Management

The MetalBackend SHALL support Apple device selection.

#### Scenario: Default device selection

- **WHEN** using MetalBackend without specifying a device
- **THEN** the default GPU is selected
- **AND** preference is given to integrated GPU on Apple Silicon

#### Scenario: Device enumeration

- **WHEN** enumerating available Metal devices
- **THEN** all GPUs are listed
- **AND** device names and memory sizes are available

### Requirement: Numerical Correctness

All Metal operations SHALL produce numerically correct results.

#### Scenario: Matmul accuracy

- **WHEN** multiplying matrices of various sizes
- **THEN** results match CpuBackend within 1e-5 relative error

#### Scenario: Sigmoid stability

- **WHEN** computing sigmoid of large values
- **THEN** results are in [0, 1] without overflow
- **AND** match the numerically stable CPU implementation

### Requirement: Error Handling

The backend SHALL handle Metal errors gracefully.

#### Scenario: Non-macOS platform

- **WHEN** compiling on non-macOS platform with `metal` feature
- **THEN** compilation fails with clear error message

#### Scenario: Out of GPU memory

- **WHEN** allocating a tensor that exceeds GPU memory
- **THEN** a clear error is returned
- **AND** error message indicates memory constraint

### Requirement: Performance

Metal operations SHALL provide speedup over CPU on Apple Silicon.

#### Scenario: Matmul performance

- **WHEN** multiplying 512x512 matrices on Apple Silicon
- **THEN** Metal is significantly faster than CpuBackend

#### Scenario: Unified memory efficiency

- **WHEN** alternating between CPU and GPU operations
- **THEN** no copy overhead is incurred
- **AND** overall throughput is high
