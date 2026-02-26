## ADDED Requirements

### Requirement: LR Scheduler trait
The system SHALL provide an `LRScheduler` trait for learning rate adjustment.

#### Scenario: Scheduler returns learning rate
- **WHEN** scheduler's `step()` method is called with epoch and metrics
- **THEN** it SHALL return the learning rate for that epoch

#### Scenario: Current LR accessible
- **WHEN** `current_lr()` is called
- **THEN** scheduler SHALL return the most recently computed learning rate

### Requirement: StepLR scheduler
The system SHALL provide `StepLR` scheduler that decays LR at fixed intervals.

#### Scenario: Decay at step size
- **WHEN** `StepLR` has initial_lr=0.1, step_size=30, gamma=0.1
- **THEN** LR SHALL be 0.1 for epochs 0-29, 0.01 for epochs 30-59, 0.001 for epochs 60+

### Requirement: ExponentialLR scheduler
The system SHALL provide `ExponentialLR` scheduler with exponential decay.

#### Scenario: Exponential decay
- **WHEN** `ExponentialLR` has initial_lr=0.1, gamma=0.9
- **THEN** LR SHALL be 0.1 * 0.9^epoch for each epoch

### Requirement: CosineAnnealingLR scheduler
The system SHALL provide `CosineAnnealingLR` scheduler with cosine annealing.

#### Scenario: Cosine annealing
- **WHEN** `CosineAnnealingLR` has initial_lr=0.1, T_max=100
- **THEN** LR SHALL follow cosine curve from initial_lr to eta_min over T_max epochs

### Requirement: ReduceLROnPlateau scheduler
The system SHALL provide `ReduceLROnPlateau` that reduces LR when metric plateaus.

#### Scenario: Reduce on plateau
- **WHEN** monitored metric hasn't improved for `patience` epochs
- **THEN** LR SHALL be multiplied by `factor`

#### Scenario: Metric from callbacks
- **WHEN** `ReduceLROnPlateau` monitors `val_loss`
- **THEN** it SHALL read metric from TrainingState metrics hashmap

### Requirement: Scheduler integration with Trainer
Schedulers SHALL be registered via `TrainerBuilder`.

#### Scenario: Register scheduler
- **WHEN** user calls `.with_lr_scheduler(scheduler)` on builder
- **THEN** scheduler SHALL adjust optimizer's learning rate each epoch

#### Scenario: Scheduler called at epoch end
- **WHEN** epoch completes
- **THEN** scheduler's `step()` SHALL be called before next epoch
