# Design: MLP Model Implementation

## Architecture

Follow the existing LinearModel pattern with type-state for training/inference separation.

### File Structure

```
lib/src/
├── model/
│   ├── mod.rs           # Add exports
│   ├── mlp.rs           # NEW: MLPModel, MLPParams, LayerParams
│   └── activation.rs    # NEW: Activation enum
├── backend/
│   ├── mod.rs           # Add relu_1d, tanh_1d, etc.
│   └── cpu.rs           # Implement new operations
├── optimizer/
│   └── mod.rs           # Add SGD<MLPParams> impl
└── examples/
    ├── train_mlp.rs     # NEW: Basic example
    └── mlp_cpu_wgpu_comparison.rs  # NEW: Benchmark
```

## Data Structures

### LayerParams

```rust
#[derive(Clone)]
pub struct LayerParams<B: Backend> {
    pub weights: Tensor2D<B>,  // (out_features, in_features)
    pub bias: Tensor1D<B>,     // (out_features,)
}
```

### MLPParams

```rust
#[derive(Clone)]
pub struct MLPParams<B: Backend> {
    pub layers: Vec<LayerParams<B>>,
}

impl<B: Backend> ParamOps<B> for MLPParams<B> {
    fn add(&self, other: &Self) -> Self;
    fn scale(&self, scalar: Scalar<B>) -> Self;
    fn l2_norm(&self) -> Scalar<B>;
}
```

### MLPModel

```rust
pub struct MLPModel<B: Backend, S> {
    params: MLPParams<B>,
    activations: Vec<Activation>,
    layer_sizes: Vec<usize>,
    _state: PhantomData<S>,
}

pub type MLP<B> = MLPModel<B, Unfitted>;
pub type MLPRegressor = MLP<CpuBackend>;
```

## Algorithms

### Forward Pass

```
Input: X (batch_size, input_features)
Cache: Store pre/post activations for backprop

for i in 0..num_layers:
    z = matmul(X, W[i]^T) + b[i]  // Linear
    a = activation[i].forward(z)   // Nonlinear
    cache.store(z, a)
    X = a

return X, cache
```

### Backward Pass

```
Input: grad_output, cache
Output: MLPParams (gradients)

delta = grad_output
grads = []

for i in (num_layers-1)..0:
    // Activation derivative
    delta = activation[i].backward(cache.post[i], delta)

    // Gradient computation
    input = if i == 0 { X } else { cache.post[i-1] }
    grad_W[i] = outer(delta, input) / batch_size
    grad_b[i] = mean(delta, axis=0)

    // Propagate
    delta = matmul(delta, W[i])

return MLPParams { layers: grads }
```

## Backend Extensions

Add to Backend trait:

```rust
// Activation functions
fn relu_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
fn relu_2d(x: &Self::Tensor2D) -> Self::Tensor2D;
fn tanh_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
fn tanh_2d(x: &Self::Tensor2D) -> Self::Tensor2D;

// Bias addition (broadcasting)
fn add_bias_per_row(t: &Self::Tensor2D, bias: &Self::Tensor1D) -> Self::Tensor2D;
```

## ONNX Export

Implement `OnnxNodeBuilder` for MLP:

```rust
impl<B: Backend> OnnxNodeBuilder for MLPModel<B, Fitted> {
    fn build_onnx_nodes(&self, graph: &mut Graph) -> Result<(), OnnxError> {
        let mut input_name = "input".to_string();

        for (i, layer) in self.params.layers.iter().enumerate() {
            // Add MatMul node
            let matmul_output = format!("matmul_{}", i);
            graph.add_node(Node {
                op_type: "MatMul".to_string(),
                inputs: vec![input_name.clone(), format!("W{}", i)],
                outputs: vec![matmul_output.clone()],
            });

            // Add Add node (bias)
            let add_output = format!("add_{}", i);
            graph.add_node(Node {
                op_type: "Add".to_string(),
                inputs: vec![matmul_output, format!("b{}", i)],
                outputs: vec![add_output.clone()],
            });

            // Add activation node
            let activation_output = format!("activation_{}", i);
            graph.add_node(Node {
                op_type: self.activations[i].onnx_op_type(),
                inputs: vec![add_output],
                outputs: vec![activation_output.clone()],
            });

            input_name = activation_output;
        }

        Ok(())
    }
}
```

## Testing Strategy

### Unit Tests

1. Activation forward/backward correctness
2. ParamOps for MLPParams
3. Single layer forward/backward
4. Multi-layer forward/backward

### Integration Tests

1. XOR problem (must learn non-linearity)
2. California Housing regression
3. ONNX export/import roundtrip

### Benchmarks

1. CPU vs WGPU training time
2. CPU vs WGPU inference time
3. Memory usage comparison
