## Context

The current `Trainer` implementation supports:
- Configurable loss, optimizer, regularizer
- Batch size and max epochs
- Early stopping and gradient clipping (opt-in)

What's missing for production use:
- No way to monitor training progress externally
- No checkpoint management (save best models, resume training)
- No learning rate adjustment during training
- No validation loop during training

This design adds a callback-based architecture that hooks into training events, enabling all these features without modifying the core training loop significantly.

## Goals / Non-Goals

**Goals:**
- Implement a `Callback` trait with hooks for training events
- Create built-in callbacks: Validation, Checkpoint, Logging
- Implement learning rate schedulers with a common `LRScheduler` trait
- Support checkpoint restoration to resume training
- Maintain backward compatibility - existing code works unchanged
- Keep the callback system simple and composable

**Non-Goals:**
- Distributed training support
- TensorBoard native format (use JSON that can be converted)
- Model versioning/beyond top-N checkpoints
- Automatic hyperparameter tuning

## Decisions

### 1. Callback Architecture

**Decision**: Use a `Callback` trait with optional methods using empty default implementations.

```rust
pub trait Callback<B: Backend, M: TrainableModel<B>> {
    fn on_train_start(&mut self, _state: &TrainingState<B, M>) {}
    fn on_train_end(&mut self, _state: &TrainingState<B, M>) {}
    fn on_epoch_start(&mut self, _state: &TrainingState<B, M>) {}
    fn on_epoch_end(&mut self, _state: &mut TrainingState<B, M>) {}
    fn on_batch_start(&mut self, _state: &TrainingState<B, M>) {}
    fn on_batch_end(&mut self, _state: &mut TrainingState<B, M>) {}
}
```

**Rationale**: Trait-based design is idiomatic Rust, allows compile-time type checking, and enables composition via callback chains. Empty defaults mean users only implement what they need.

**Alternatives considered**:
- Function pointers: Less flexible, no state
- Enum-based: Requires defining all variants upfront
- Closure-based: Can't store state easily

### 2. TrainingState Structure

**Decision**: Pass mutable/immutable state to callbacks via a `TrainingState` struct.

```rust
pub struct TrainingState<'a, B: Backend, M: TrainableModel<B>> {
    pub epoch: usize,
    pub batch: usize,
    pub total_epochs: usize,
    pub total_batches: usize,
    pub loss: f64,
    pub model: &'a M,
    pub params: &'a M::Params,
    pub gradients: Option<&'a M::Params>,
    pub learning_rate: f64,
    pub metrics: HashMap<String, f64>,
    pub stop_requested: bool,  // Set by callbacks to halt training
}
```

**Rationale**: Centralizes all training information in one place. Mutable access on `on_epoch_end` allows callbacks to modify state (like stopping training).

### 3. Learning Rate Scheduler Design

**Decision**: Create an `LRScheduler` trait that returns the current learning rate given the epoch.

```rust
pub trait LRScheduler {
    fn step(&mut self, epoch: usize, metrics: &HashMap<String, f64>) -> f64;
    fn current_lr(&self) -> f64;
}
```

**Implementations**:
- `StepLR`: Decay by gamma every step_size epochs
- `ExponentialLR`: Decay by gamma every epoch
- `CosineAnnealingLR`: Cosine annealing schedule
- `ReduceLROnPlateau`: Reduce when metric plateaus (needs callback integration)

**Rationale**: Simple trait that can be called at epoch boundaries. ReduceLROnPlateau integrates with callbacks to access validation metrics.

### 4. Checkpoint Format

**Decision**: Use a directory-based checkpoint format with metadata JSON and model binary.

```
checkpoints/
├── checkpoint_epoch_001.json
├── checkpoint_epoch_001.bin
├── checkpoint_epoch_005.json
├── checkpoint_epoch_005.bin
├── best/
│   ├── best_epoch_003.json
│   └── best_epoch_003.bin
```

**Metadata JSON**:
```json
{
  "epoch": 5,
  "loss": 0.0234,
  "metrics": {"val_loss": 0.0312, "val_accuracy": 0.89},
  "learning_rate": 0.001,
  "timestamp": "2024-02-25T10:30:00Z"
}
```

**Rationale**: Separating metadata as JSON allows inspection without loading the model. Directory structure makes it easy to find best/latest checkpoints.

**Alternatives considered**:
- Single file: Harder to inspect, requires parsing entire file
- Database: Overkill for this use case

### 5. Validation Callback Integration

**Decision**: `ValidationCallback` takes a separate validation dataset and runs full evaluation every N epochs.

```rust
pub struct ValidationCallback<B: Backend, M, L: Loss<B>>
where
    M: TrainableModel<B> + InferenceModel<B>,
{
    val_dataset: InMemoryDataset,
    loss_fn: L,
    frequency: usize,  // Run every N epochs
    batch_size: usize,
}
```

**Rationale**: Separate dataset and loss allows different validation metrics than training. Frequency parameter gives control over validation overhead.

## Risks / Trade-offs

### Risk: Callback Performance Overhead
Calling callbacks at every batch/epoch adds overhead.
**Mitigation**: Empty callbacks are no-ops with minimal overhead. Document that heavy work should use frequency parameters.

### Risk: Mutable State in Callbacks
Callbacks with mutable state could cause issues if called concurrently (future concern).
**Mitigation**: Document that callbacks are called sequentially. Current design is single-threaded.

### Trade-off: No Async Callbacks
Callbacks are synchronous, blocking the training loop.
**Mitigation**: Document that long-running operations (network logging) should spawn threads or use channels.

### Risk: Checkpoint Disk Space
Saving many checkpoints could fill disk.
**Mitigation**: `CheckpointCallback` has `keep_best_n` parameter to limit checkpoints.

## Migration Plan

1. Add `callbacks` and `schedulers` modules with traits and implementations
2. Extend `TrainerBuilder` with `.with_callback()` and `.with_lr_scheduler()` methods
3. Modify `Trainer::fit()` to invoke callbacks at appropriate points
4. All features are opt-in via builder - no breaking changes
5. Add examples demonstrating each feature
