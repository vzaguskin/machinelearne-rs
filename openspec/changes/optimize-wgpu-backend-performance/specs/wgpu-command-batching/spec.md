## ADDED Requirements

### Requirement: Command Buffer Accumulation

The WGPU backend SHALL accumulate multiple operations into a single command buffer before GPU submission.

#### Scenario: Operations batched automatically

- **WHEN** multiple tensor operations are called in sequence (add, mul, matvec, etc.)
- **THEN** operations are queued in a command accumulator
- **AND** no GPU submission occurs until flush
- **AND** all queued operations execute in single submission

#### Scenario: Automatic flush on data read

- **WHEN** `to_vec()` is called on a tensor
- **THEN** the command accumulator is automatically flushed
- **AND** all pending operations complete before data is returned

#### Scenario: Manual flush control

- **WHEN** `WgpuDevice::flush()` is called
- **THEN** all pending operations are submitted to GPU
- **AND** execution blocks until completion

### Requirement: Transparent API Compatibility

Command batching SHALL be transparent to users of the Backend trait.

#### Scenario: Existing code works unchanged

- **WHEN** existing code uses WgpuBackend through Backend trait methods
- **THEN** operations execute correctly with batching
- **AND** no code changes are required
- **AND** results are numerically identical to non-batched execution

### Requirement: Flush Threshold

The backend SHALL auto-flush when accumulated operations exceed a threshold.

#### Scenario: Auto-flush threshold reached

- **WHEN** more than N operations are queued (N configurable, default 50)
- **THEN** command buffer is automatically submitted
- **AND** subsequent operations start new accumulator
