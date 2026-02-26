## ADDED Requirements

### Requirement: Checkpoint loading
The system SHALL provide functionality to load checkpoints for training resumption.

#### Scenario: Load checkpoint by path
- **WHEN** user calls checkpoint load function with path
- **THEN** model parameters SHALL be restored from checkpoint

#### Scenario: Load checkpoint metadata
- **WHEN** checkpoint is loaded
- **THEN** metadata (epoch, loss, metrics) SHALL be accessible

### Requirement: Training resumption
Training SHALL be resumable from a loaded checkpoint.

#### Scenario: Resume from epoch
- **WHEN** training is resumed from checkpoint saved at epoch 50
- **THEN** training SHALL continue from epoch 51

#### Scenario: Resume with optimizer state
- **WHEN** checkpoint includes optimizer state
- **THEN** optimizer state (momentum for Adam) SHALL be restored

### Requirement: Find latest checkpoint
The system SHALL provide utility to find the latest checkpoint.

#### Scenario: Find latest in directory
- **WHEN** `find_latest_checkpoint(dir)` is called
- **THEN** path to most recent checkpoint SHALL be returned

### Requirement: Find best checkpoint
The system SHALL provide utility to find the best checkpoint.

#### Scenario: Find best by metric
- **WHEN** `find_best_checkpoint(dir, "val_loss", mode="min")` is called
- **THEN** path to checkpoint with lowest val_loss SHALL be returned
