## Context

The current ONNX export implementation has these issues:
- `OnnxExportable` is implemented only for `LinearModel<B, Fitted>` with methods that return bytes directly
- Preprocessing export in `from_pipeline.rs` uses a large `match` statement with hardcoded export logic
- No abstraction for models or transformers to contribute nodes to the graph
- Adding new exportable types requires modifying library code

## Goals / Non-Goals

**Goals:**
- Create a `OnnxNodeBuilder` trait for transformers to contribute ONNX nodes
- Refactor `OnnxExportable` to work with `OnnxGraphBuilder` context
- Enable third-party types to implement ONNX export
- Maintain backward compatibility for basic use cases (`save_onnx()`, `to_onnx()`)

**Non-Goals:**
- Adding new ONNX operators (use existing ones)
- Supporting ONNX import/parsing (only export)
- Performance optimization of export (current performance is acceptable)

## Decisions

### 1. Trait Hierarchy

**Decision**: Use two traits: `OnnxNodeBuilder` for graph contributions and `OnnxExportable` for final model export.

```rust
/// For types that can contribute nodes to an ONNX graph
pub trait OnnxNodeBuilder {
    /// Add nodes to the graph, return output tensor name
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError>;
}

/// For types that can be exported as complete ONNX models
pub trait OnnxExportable {
    fn build_onnx_graph(&self, builder: &mut OnnxGraphBuilder) -> Result<String, OnnxError>;

    // Convenience methods (default implementations)
    fn to_onnx(&self) -> Result<Vec<u8>, OnnxError> { ... }
    fn save_onnx(&self, path: impl AsRef<Path>) -> Result<(), OnnxError> { ... }
}
```

**Rationale**: Separating node building from model export allows:
- Transformers to be composed in pipelines
- Models to focus on their core logic
- Either trait to be implemented independently

**Alternatives considered**:
- Single trait with optional methods: More confusing, harder to implement correctly
- No traits, just functions: Less extensible, can't add new types

### 2. OnnxGraphBuilder Enhancement

**Decision**: Enhance `OnnxGraphBuilder` to track:
- Current input/output tensor names
- Tensor shapes for validation
- Name uniquification

```rust
impl OnnxGraphBuilder {
    pub fn add_node(&mut self, node: NodeProto) -> Result<String, OnnxError>;
    pub fn add_initializer(&mut self, name: &str, tensor: TensorProto) -> Result<(), OnnxError>;
    pub fn unique_name(&self, prefix: &str) -> String;
    pub fn validate_shape(&self, tensor_name: &str, expected: &[i64]) -> Result<(), OnnxError>;
}
```

**Rationale**: Centralizing these concerns in the builder reduces boilerplate in implementations.

### 3. Transformer Export Migration

**Decision**: Each fitted transformer type implements `OnnxNodeBuilder`.

```rust
impl OnnxNodeBuilder for FittedStandardScaler {
    fn build_onnx_nodes(&self, builder: &mut OnnxGraphBuilder, input_name: &str) -> Result<String, OnnxError> {
        // Add Sub node for mean, Div node for scale
        // Return output tensor name
    }
}
```

**Rationale**: Trait-based dispatch replaces the hardcoded match statement, enabling extensibility.

**Migration path**: Keep the existing `export_preproc_step` function that delegates to the trait, then deprecate it.

### 4. Model Export Pattern

**Decision**: Models implement `OnnxExportable` using `OnnxNodeBuilder` for their internals.

```rust
impl<B: Backend> OnnxExportable for LinearModel<B, Fitted> {
    fn build_onnx_graph(&self, builder: &mut OnnxGraphBuilder) -> Result<String, OnnxError> {
        // Add weights/bias as initializers
        // Add Gemm node
        // Return output tensor name
    }
}
```

**Rationale**: Models can use the same node-building primitives as transformers.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Breaking existing implementors | Provide default implementations for convenience methods |
| Complex trait bounds | Use concrete types where possible, avoid over-genericization |
| Performance overhead | Builder pattern has minimal overhead, only allocates during export |
| Incomplete ONNX coverage | Document supported operators, fail gracefully for unsupported |

## Migration Plan

1. Add `OnnxNodeBuilder` trait with implementations for existing transformers
2. Refactor `OnnxExportable` to use `build_onnx_graph` internally
3. Update `FittedPipeline` to use trait-based dispatch
4. Add deprecation warnings to old functions
5. Remove old functions in next major version

No rollback needed - old API can coexist with new during migration.
