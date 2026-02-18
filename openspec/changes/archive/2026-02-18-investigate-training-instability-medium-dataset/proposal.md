## Why

Training loss explodes to infinity on medium-sized datasets (10K samples) while the same hyperparameters work correctly for small (1K) and large (20K) datasets. This instability affects both CPU and WGPU backends, indicating a fundamental issue in the training pipeline rather than a backend-specific bug. Users cannot reliably train models on certain dataset sizes without manual hyperparameter tuning.

## What Changes

- Add gradient clipping to prevent gradient explosion during training
- Implement convergence monitoring to detect when training diverges
- Add early stopping mechanism to halt training when loss stops improving or diverges
- Improve default hyperparameters for better stability across dataset sizes
- Add diagnostic logging for gradient magnitudes during training

## Capabilities

### New Capabilities

- `gradient-clipping`: Capability to clip gradients during training to prevent explosion
- `early-stopping`: Capability to halt training when convergence criteria are met or divergence is detected

### Modified Capabilities

- `model-training`: Add optional gradient clipping and early stopping to the training loop. The trainer should support configuration of stability features without breaking existing API.
- `sgd-optimizer`: May need to consider gradient scaling or adaptive learning rate adjustments for better stability.

## Impact

**Affected Code:**
- `lib/src/trainer/mod.rs` - Core training loop modifications
- `lib/src/optimizer/mod.rs` - Potential gradient clipping integration
- `lib/src/loss/mod.rs` - May need gradient magnitude utilities

**API Changes:**
- `TrainerBuilder` will gain new optional methods: `.gradient_clipping()`, `.early_stopping()`
- Backward compatible - existing code continues to work

**Dependencies:**
- No new external dependencies required

**Examples:**
- May need to update examples to demonstrate stability features
- `lib/examples/wgpu_cpu_comparison.rs` should benefit from these fixes
