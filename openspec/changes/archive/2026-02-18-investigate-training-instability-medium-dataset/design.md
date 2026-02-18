## Context

The training loop in `lib/src/trainer/mod.rs` currently lacks stability mechanisms. When training on certain dataset sizes (notably 10K samples), gradients can explode causing loss to diverge to infinity. This is not backend-specific - both CPU and WGPU exhibit the same behavior.

Current training flow:
1. Forward pass → predictions
2. Loss computation + regularizer penalty
3. Gradient computation (loss + regularizer)
4. Optimizer step: `params = params - lr * gradients`
5. Repeat for max_epochs

**Missing mechanisms:**
- No gradient magnitude limiting
- No divergence detection
- No early stopping when training plateaus or diverges

## Goals / Non-Goals

**Goals:**
- Add gradient clipping to prevent gradient explosion
- Add early stopping to halt training when loss diverges or plateaus
- Maintain backward compatibility - existing code works without changes
- Keep trainer API fluent and ergonomic

**Non-Goals:**
- Adaptive learning rates (would require optimizer redesign)
- Complex convergence criteria (keep it simple: loss-based)
- Gradient clipping per-parameter (clip by global norm only)

## Decisions

### Decision 1: Gradient Clipping Strategy

**Choice:** Clip gradients by global L2 norm (max_norm)

**Rationale:**
- Most common approach in ML frameworks (PyTorch, TensorFlow)
- Preserves gradient direction while limiting magnitude
- Single threshold parameter is intuitive

**Alternatives considered:**
- **Clip by value:** Simpler but can change gradient direction. Rejected.
- **Clip by global norm + rescale:** Current choice. Standard approach.

**Implementation:**
```rust
// In TrainerBuilder
pub fn gradient_clipping(mut self, max_norm: f32) -> Self {
    self.gradient_clipping = Some(max_norm);
    self
}
```

During training, after computing `total_grads`, compute L2 norm and rescale if needed:
```rust
if let Some(max_norm) = self.gradient_clipping {
    total_grads = clip_by_norm(total_grads, max_norm);
}
```

### Decision 2: Early Stopping Strategy

**Choice:** Patience-based early stopping monitoring loss improvement

**Rationale:**
- Industry standard approach
- Prevents overfitting and detects divergence
- Simple to configure and understand

**Configuration:**
- `patience`: Number of epochs without improvement before stopping
- `min_delta`: Minimum change to qualify as improvement
- `divergence_threshold`: Loss ratio that indicates divergence (e.g., 10x increase)

**Alternatives considered:**
- **Fixed epoch count only:** Current behavior. Can waste time or diverge.
- **Loss threshold only:** Doesn't handle plateaus well.
- **Patience + divergence detection:** Current choice. Comprehensive.

**Implementation:**
```rust
// In TrainerBuilder
pub fn early_stopping(mut self, patience: usize, min_delta: f32) -> Self {
    self.early_stopping = Some(EarlyStoppingConfig { patience, min_delta });
    self
}

pub fn divergence_threshold(mut self, threshold: f32) -> Self {
    self.divergence_threshold = Some(threshold);
    self
}
```

### Decision 3: Where to Implement

**Choice:** Implement in `Trainer` and `TrainerBuilder`, not in optimizer

**Rationale:**
- Gradient clipping is training-loop concern, not optimizer concern
- Early stopping requires epoch-level state, fits naturally in trainer
- Keeps optimizer focused on parameter updates only
- Consistent with separation of concerns (ADR-0001)

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Gradient clipping may slow convergence on well-behaved data | Make it opt-in; default is no clipping |
| Early stopping patience tuning is dataset-dependent | Document reasonable defaults; provide guidance |
| Clipping norm computation adds overhead | Only compute when clipping is enabled |
| False positive early stopping | Allow users to disable via `None` or high patience |

## Migration Plan

No migration needed. Changes are additive:
- New optional methods on `TrainerBuilder`
- Default behavior unchanged (no clipping, no early stopping)
- Existing code compiles and behaves identically

**Recommended migration for affected users:**
```rust
// Before: Could diverge on 10K dataset
let trainer = Trainer::builder(loss, optimizer, regularizer)
    .max_epochs(50)
    .build();

// After: Stable training
let trainer = Trainer::builder(loss, optimizer, regularizer)
    .max_epochs(50)
    .gradient_clipping(1.0)           // Add clipping
    .early_stopping(5, 0.001)         // Add early stopping
    .divergence_threshold(10.0)       // Detect divergence
    .build();
```
