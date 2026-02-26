## ADDED Requirements

### Requirement: Staging buffer pooling
The system SHALL pool and reuse staging buffers for CPU readback operations.

#### Scenario: Staging buffer reuse
- **WHEN** `to_vec()` is called multiple times
- **THEN** staging buffers are reused from a pool instead of being allocated each time

#### Scenario: Pool size limit
- **WHEN** the staging buffer pool exceeds its maximum size
- **THEN** least-recently-used buffers are deallocated

### Requirement: Efficient data readback
The system SHALL minimize CPU↔GPU synchronization overhead during data readback.

#### Scenario: Single sync for multiple reads
- **WHEN** multiple tensors need to be read back to CPU
- **THEN** they share a single GPU synchronization point when possible

#### Scenario: Async buffer mapping
- **WHEN** reading data from GPU to CPU
- **THEN** buffer mapping is initiated before blocking wait to overlap operations
