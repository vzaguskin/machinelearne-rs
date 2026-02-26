## Why

The current training loop lacks production-ready features needed for real ML workflows: no way to monitor training progress, save/restore checkpoints, adjust learning rates during training, or run validation at intervals. Users must manually implement these patterns, leading to duplicated effort and inconsistent implementations.

This change brings the trainer up to par with frameworks like PyTorch (callbacks, LR schedulers) and TensorFlow (TensorBoard logging, checkpoint management), enabling reproducible experiments and fault-tolerant training.

## What Changes

- **Training Callbacks System**: Hook-based API to inject custom behavior at key training events (epoch start/end, batch start/end, validation)
- **Validation Callback**: Built-in callback to run validation every N epochs with configurable metrics
- **Checkpoint Callback**: Save top-N best models based on validation metric, with configurable checkpoint directory
- **Logging Callback**: Structured logging interface with JSON output (TensorBoard/W&B compatible format)
- **Learning Rate Schedulers**: StepLR, ExponentialLR, CosineAnnealingLR, and ReduceLROnPlateau schedulers
- **Checkpoint Restoration**: Load trainer state (model params, optimizer state, epoch) from checkpoint to resume training

## Capabilities

### New Capabilities

- `training-callbacks`: Hook-based callback system for training events (on_epoch_start, on_epoch_end, on_batch_start, on_batch_end, on_train_start, on_train_end)
- `validation-callback`: Configurable validation runner that evaluates on a separate dataset every N epochs
- `checkpoint-callback`: Model checkpointing that saves best-N models based on a monitored metric
- `training-logging`: Structured logging with JSON output for integration with visualization tools
- `lr-schedulers`: Learning rate schedulers (StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau)
- `checkpoint-restore`: Ability to resume training from a saved checkpoint

### Modified Capabilities

- `model-training`: Extend trainer to support callback registration and learning rate scheduler integration

## Impact

- **Trainer API**: New `TrainerBuilder` methods: `.with_callback()`, `.with_lr_scheduler()`, `.with_validation()`
- **New Modules**: `lib/src/callbacks/`, `lib/src/schedulers/`
- **Checkpoint Format**: Binary format using existing serialization (bincode) with metadata JSON
- **Breaking Changes**: None - all features are opt-in via builder pattern
