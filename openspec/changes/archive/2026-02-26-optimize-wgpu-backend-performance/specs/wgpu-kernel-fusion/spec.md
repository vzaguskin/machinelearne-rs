## ADDED Requirements

### Requirement: Fused Forward Pass

The WGPU backend SHALL provide a fused kernel for linear layer forward pass.

#### Scenario: Fused matvec + bias

- **WHEN** computing `y = W @ x + b` for a linear layer
- **THEN** a single fused kernel executes both operations
- **AND** no intermediate buffer is created
- **AND** single kernel launch instead of two

#### Scenario: Fused kernel used automatically

- **WHEN** forward pass is called during training
- **THEN** the fused forward kernel is used if available
- **AND** falls back to separate operations if dimensions don't match fusion criteria

### Requirement: Fused Backward Pass

The WGPU backend SHALL provide a fused kernel for linear layer backward pass.

#### Scenario: Fused gradient computation

- **WHEN** computing gradients for linear layer during backpropagation
- **THEN** weight gradients and bias gradients are computed in single kernel
- **AND** reduces kernel launches by 50%

### Requirement: Fusion Fallback

Fused kernels SHALL gracefully fall back to individual operations when needed.

#### Scenario: Unsupported dimensions

- **WHEN** tensor dimensions don't match fused kernel requirements
- **THEN** individual operations are used instead
- **AND** results remain correct
