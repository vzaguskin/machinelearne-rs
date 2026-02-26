## 1. Core Callback Infrastructure

- [x] 1.1 Create `lib/src/callbacks/mod.rs` with `Callback` trait definition
- [x] 1.2 Create `TrainingState` struct with epoch, batch, loss, metrics, etc.
- [x] 1.3 Implement `Callback` trait with empty default implementations for all hooks
- [x] 1.4 Add `callbacks` module export to `lib/src/lib.rs`

## 2. Trainer Integration for Callbacks

- [x] 2.1 Add `callbacks: Vec<Box<dyn Callback>>` field to `Trainer` struct
- [x] 2.2 Add `with_callback()` method to `TrainerBuilder`
- [x] 2.3 Modify `Trainer::fit()` to invoke callbacks at training events
- [x] 2.4 Implement `stop_requested` flag check in training loop
- [x] 2.5 Ensure backward compatibility when no callbacks registered

## 3. Learning Rate Schedulers

- [x] 3.1 Create `lib/src/schedulers/mod.rs` with `LRScheduler` trait
- [x] 3.2 Implement `StepLR` scheduler (decay every N epochs)
- [x] 3.3 Implement `ExponentialLR` scheduler (decay by gamma each epoch)
- [x] 3.4 Implement `CosineAnnealingLR` scheduler (cosine curve)
- [x] 3.5 Implement `ReduceLROnPlateau` scheduler (reduce on metric plateau)
- [x] 3.6 Add `schedulers` module export to `lib/src/lib.rs`

## 4. Trainer Integration for Schedulers

- [x] 4.1 Add `lr_scheduler: Option<Box<dyn LRScheduler>>` field to `Trainer`
- [x] 4.2 Add `with_lr_scheduler()` method to `TrainerBuilder`
- [x] 4.3 Modify training loop to call scheduler at epoch end
- [x] 4.4 Update optimizer's learning rate from scheduler each epoch

## 5. Validation Callback

- [x] 5.1 Create `lib/src/callbacks/validation.rs` with `ValidationCallback` struct
- [x] 5.2 Implement validation loop that runs on separate dataset
- [x] 5.3 Compute validation loss and store in metrics
- [x] 5.4 Support configurable validation frequency (every N epochs)
- [x] 5.5 Implement `Callback` trait for `ValidationCallback`

## 6. Checkpoint Callback

- [x] 6.1 Create `lib/src/callbacks/checkpoint.rs` with `CheckpointCallback` struct
- [x] 6.2 Implement checkpoint saving with `.bin` and `.json` files
- [x] 6.3 Add metadata JSON serialization (epoch, loss, metrics, lr, timestamp)
- [x] 6.4 Implement metric-based saving (save when metric improves)
- [x] 6.5 Implement best-N retention (keep only top N checkpoints)
- [x] 6.6 Implement periodic saving (every N epochs)
- [x] 6.7 Implement `Callback` trait for `CheckpointCallback`

## 7. Logging Callback

- [x] 7.1 Create `lib/src/callbacks/logging.rs` with `LoggingCallback` struct
- [x] 7.2 Implement JSON Lines format logging
- [x] 7.3 Log training events (epoch, batch, loss, lr, metrics)
- [x] 7.4 Implement file output option
- [x] 7.5 Implement console output option
- [x] 7.6 Implement log flushing at epoch/training end
- [x] 7.7 Implement `Callback` trait for `LoggingCallback`

## 8. Checkpoint Restoration

- [x] 8.1 Create `lib/src/checkpoint/mod.rs` with checkpoint utilities
- [x] 8.2 Implement `load_checkpoint()` function to restore model params
- [x] 8.3 Implement `CheckpointMetadata` struct for parsing JSON
- [x] 8.4 Implement `find_latest_checkpoint()` utility function
- [x] 8.5 Implement `find_best_checkpoint()` utility function
- [x] 8.6 Add `resume_from_checkpoint()` method to `TrainerBuilder`

## 9. Unit Tests

- [x] 9.1 Test Callback trait default implementations
- [x] 9.2 Test TrainingState creation and mutation
- [x] 9.3 Test callback registration and invocation order
- [x] 9.4 Test stop_requested flag
- [x] 9.5 Test StepLR scheduler math
- [x] 9.6 Test ExponentialLR scheduler math
- [x] 9.7 Test CosineAnnealingLR scheduler math
- [x] 9.8 Test ReduceLROnPlateau scheduler
- [x] 9.9 Test ValidationCallback metrics computation
- [x] 9.10 Test CheckpointCallback file creation
- [x] 9.11 Test LoggingCallback output format
- [x] 9.12 Test checkpoint loading and restoration

## 10. Documentation and Examples

- [x] 10.1 Add rustdoc to Callback trait and TrainingState
- [x] 10.2 Add rustdoc to all scheduler types
- [x] 10.3 Add rustdoc to all callback types
- [x] 10.4 Create example: `train_mlp_with_callbacks.rs`
- [x] 10.5 Create example: `train_with_lr_scheduling.rs`
- [x] 10.6 Create example: `train_with_checkpoints.rs`
- [x] 10.7 Update CHANGELOG.md

## 11. Verification

- [x] 11.1 Run `cargo test -p machinelearne-rs`
- [x] 11.2 Run `cargo fmt` and `cargo clippy`
- [x] 11.3 Verify coverage threshold met
- [x] 11.4 Verify backward compatibility (existing tests pass)
