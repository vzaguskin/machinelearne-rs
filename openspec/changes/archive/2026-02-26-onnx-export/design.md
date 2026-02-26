# Design: ONNX Export Implementation

## Architecture

```
lib/src/onnx/
├── mod.rs              # Module exports, OnnxError
├── proto.rs            # ONNX protobuf type definitions
├── graph.rs            # OnnxGraphBuilder
├── operators.rs        # Node creation helpers
├── export.rs           # OnnxExportable trait
├── from_pipeline.rs    # Model/transformer export implementations
└── inference.rs        # OnnxInferenceSession (experimental)
```

## Component Details

### proto.rs - Protobuf Types

Defines minimal ONNX protobuf types using `prost`:
- `ModelProto` - Top-level container
- `GraphProto` - Computation graph
- `NodeProto` - Operator nodes
- `TensorProto` - Tensor data (initializers)
- `AttributeProto` - Node attributes

Uses prost derive macros for serialization.

### graph.rs - OnnxGraphBuilder

Builder pattern for constructing ONNX models:

```rust
pub struct OnnxGraphBuilder {
    pub model: ModelProto,
    pub graph: GraphProto,
    name_counter: HashMap<String, usize>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxGraphBuilder {
    pub fn new(model_name: &str) -> Self;
    pub fn with_ml_opset(self) -> Self;
    pub fn add_input_float(&mut self, name: &str, num_features: usize) -> &mut Self;
    pub fn add_output_float(&mut self, name: &str, num_features: usize) -> &mut Self;
    pub fn add_node(&mut self, op_type: &str, inputs: Vec<String>, outputs: Vec<String>, attrs: Vec<AttributeProto>) -> &mut Self;
    pub fn add_float_initializer(&mut self, name: &str, dims: &[i64], data: &[f32]) -> &mut Self;
    pub fn build(self) -> Result<Vec<u8>, OnnxError>;
}
```

### operators.rs - Operator Helpers

Functions for creating common ONNX operators:

| Function | ONNX Op | Purpose |
|----------|---------|---------|
| `gemm()` | Gemm | Matrix multiply (linear layer) |
| `add/sub/mul/div()` | Add/Sub/Mul/Div | Element-wise ops |
| `cast()` | Cast | Type conversion |
| `sqrt()` | Sqrt | Square root |
| `reduce_sum()` | ReduceSum | Sum reduction |

### from_pipeline.rs - Export Implementations

**LinearModel export:**
1. Extract weights and bias
2. Create initializers for weights [1, n_features] and bias [1]
3. Add Gemm node: output = input @ weights^T + bias
4. transB=1 for weight transpose

**Transformer exports:**

| Transformer | ONNX Implementation |
|-------------|---------------------|
| StandardScaler | Sub(mean) + Div(std) |
| MinMaxScaler | Sub(min) + Mul(scale) + Add(target_min) |
| RobustScaler | Sub(center) + Div(scale) |
| MaxAbsScaler | Mul(scale) |
| Normalizer | Mul^2 + ReduceSum + Sqrt + Div |
| SimpleImputer | Equal + Where (NaN replacement) |

## ONNX Graph Structure

Example: LinearModel with 3 features

```
Inputs:
  input: [batch_size, 3]

Initializers:
  weights: [1, 3] (const)
  bias: [1] (const)

Nodes:
  Gemm(input, weights, bias) -> output
    alpha=1, beta=1, transB=1

Outputs:
  output: [batch_size, 1]
```

## Feature Flags

```toml
[features]
onnx = ["dep:prost", "dep:prost-types", "dep:prost-build"]
onnx-inference = ["onnx", "dep:ort", "dep:ndarray"]
```

## Testing Strategy

1. **Unit tests** - Validate ONNX graph structure
2. **Round-trip tests** - Export, verify bytes are valid protobuf
3. **Integration tests** - Compare native vs exported predictions

## Dependencies

```toml
[dependencies]
prost = { version = "0.13", optional = true }
prost-types = { version = "0.13", optional = true }
ort = { version = "2.0.0-rc.11", optional = true }

[build-dependencies]
prost-build = { version = "0.13", optional = true }
```
