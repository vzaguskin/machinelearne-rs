## ADDED Requirements

### Requirement: Logging callback configuration
The system SHALL provide a `LoggingCallback` for structured training output.

#### Scenario: Create logging callback
- **WHEN** user creates `LoggingCallback` with output path
- **THEN** training logs SHALL be written to that path

#### Scenario: Console logging
- **WHEN** user enables console output
- **THEN** training progress SHALL be printed to stdout

### Requirement: JSON log format
Logs SHALL be written in JSON Lines format for tool compatibility.

#### Scenario: JSON log entries
- **WHEN** training events occur
- **THEN** logs SHALL be written as one JSON object per line

#### Scenario: Log entry structure
- **WHEN** log entry is written
- **THEN** it SHALL contain epoch, batch, loss, learning_rate, metrics, timestamp

### Requirement: TensorBoard-compatible output
Logs SHALL be convertible to TensorBoard format.

#### Scenario: Scalar logging
- **WHEN** metrics are logged
- **THEN** scalar values SHALL be written with step (epoch) and wall_time

### Requirement: Log flushing
Logs SHALL be periodically flushed to disk.

#### Scenario: End-of-epoch flush
- **WHEN** epoch completes
- **THEN** log buffer SHALL be flushed to disk

#### Scenario: Training end flush
- **WHEN** training completes or stops
- **THEN** all pending logs SHALL be flushed
