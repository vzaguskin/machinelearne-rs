## ADDED Requirements

### Requirement: Checkpoint callback configuration
The system SHALL provide a `CheckpointCallback` for saving model checkpoints.

#### Scenario: Create checkpoint callback
- **WHEN** user creates `CheckpointCallback` with directory and monitored metric
- **THEN** checkpoints SHALL be saved to that directory

#### Scenario: Configure best-N retention
- **WHEN** user sets `keep_best_n=3`
- **THEN** only top 3 checkpoints by monitored metric SHALL be retained

### Requirement: Metric-based checkpointing
Checkpoints SHALL be saved based on monitored metric improvement.

#### Scenario: Monitor validation loss
- **WHEN** monitored metric is `val_loss` and mode is `min`
- **THEN** checkpoint SHALL be saved when `val_loss` decreases

#### Scenario: Monitor validation accuracy
- **WHEN** monitored metric is `val_accuracy` and mode is `max`
- **THEN** checkpoint SHALL be saved when `val_accuracy` increases

### Requirement: Checkpoint file format
Checkpoints SHALL be saved in a structured format with metadata.

#### Scenario: Checkpoint files created
- **WHEN** checkpoint is saved
- **THEN** a `.bin` file with model parameters SHALL be created
- **AND** a `.json` file with metadata SHALL be created

#### Scenario: Metadata includes training state
- **WHEN** checkpoint is saved
- **THEN** metadata SHALL include epoch, loss, metrics, learning rate, timestamp

### Requirement: Periodic checkpointing
Checkpoints SHALL be saved at configurable intervals regardless of metric.

#### Scenario: Save every N epochs
- **WHEN** `save_every=10` is configured
- **THEN** checkpoint SHALL be saved every 10 epochs

### Requirement: Best checkpoint tracking
The system SHALL track the best checkpoint based on monitored metric.

#### Scenario: Best checkpoint directory
- **WHEN** checkpoints are saved
- **THEN** best checkpoint SHALL be saved to `best/` subdirectory

#### Scenario: Best checkpoint updated
- **WHEN** new best metric is achieved
- **THEN** previous best checkpoint SHALL be replaced
