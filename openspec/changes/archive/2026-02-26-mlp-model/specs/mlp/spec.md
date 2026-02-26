# Spec: MLP Model

## Overview

Multi-Layer Perceptron (MLP) is a feedforward neural network with configurable layers and activation functions.

## API

### Model Creation

```rust
// Create MLP with layer sizes and activations
let model = MLP::<CpuBackend>::new(
    &[input_size, hidden1, hidden2, output_size],
    &[Activation::ReLU, Activation::ReLU, Activation::Identity]
);

// Convenience constructors
let regressor = MLPRegressor::new(&[8, 16, 8, 1]); // ReLU hidden, Identity output
let classifier = MLPClassifier::new(&[features, 32, 16, num_classes]); // Sigmoid output
```

### Activation Functions

```rust
pub enum Activation {
    ReLU,      // max(0, x) - hidden layers
    Sigmoid,   // 1 / (1 + exp(-x)) - binary classification output
    Tanh,      // tanh(x) - hidden layers
    Identity,  // x - regression output
}
```

### Training

```rust
// Uses existing Trainer infrastructure
let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
    .max_epochs(1000)
    .gradient_clipping(1.0)  // Recommended for MLPs
    .build();

let fitted = trainer.fit(model, &dataset)?;
```

### Inference

```rust
// Single prediction
let output = fitted.predict(&input_1d);

// Batch prediction
let outputs = fitted.predict_batch(&inputs_2d);
```

### Serialization

```rust
// Save/load
fitted.save_to_file("mlp_model.bin")?;
let loaded = MLPModel::<CpuBackend, Fitted>::load_from_file("mlp_model.bin")?;
```

### ONNX Export

```rust
// Export to ONNX format
fitted.save_onnx("mlp_model.onnx", "MLPRegressor")?;
```

## Behavior

### Forward Pass

For each layer `i`:
1. Compute linear transform: `z = x @ W[i]^T + b[i]`
2. Apply activation: `a = activation[i](z)`
3. Output becomes input to next layer

### Backward Pass

Backpropagation through all layers:
1. Start with gradient from loss function
2. For each layer (reverse order):
   - Apply activation derivative
   - Compute weight/bias gradients
   - Propagate gradient to previous layer

### Initialization

- Weights: Xavier/Glorot initialization (scale = sqrt(2 / (fan_in + fan_out)))
- Biases: Zero initialization

## Constraints

- Minimum 2 layers (input → output)
- All layers are fully connected (Dense)
- Batch size must be > 0
- Input features must match first layer size

## Error Cases

- Empty layer configuration → Error
- Mismatched activation count → Error
- Input dimension mismatch → Runtime panic (assertion)
- Serialization failure → Error
