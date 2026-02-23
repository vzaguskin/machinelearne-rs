## ADDED Requirements

### Requirement: Lazy operation execution
The WGPU backend SHALL queue tensor operations for deferred execution instead of executing immediately.

#### Scenario: Operations do not block
- **WHEN** a tensor operation (add, mul, matvec, etc.) is called
- **THEN** the operation is queued and control returns immediately without waiting for GPU completion

#### Scenario: Operations execute on data read
- **WHEN** user calls `to_vec()` or `sum()` on a tensor
- **THEN** all queued operations are flushed and executed before the data is returned

### Requirement: Command accumulation threshold
The system SHALL accumulate commands up to a configurable threshold before auto-flushing.

#### Scenario: Threshold-based flush
- **WHEN** the number of queued operations exceeds the threshold (default 500)
- **THEN** operations are automatically flushed to GPU

#### Scenario: Manual flush
- **WHEN** `device.flush()` is called explicitly
- **THEN** all queued operations are submitted to GPU immediately

### Requirement: Compute pass bundling
The system SHALL bundle compatible operations into single compute passes to reduce driver overhead.

#### Scenario: Multiple operations bundled
- **WHEN** multiple tensor operations are queued sequentially
- **THEN** they are executed within minimal compute passes
