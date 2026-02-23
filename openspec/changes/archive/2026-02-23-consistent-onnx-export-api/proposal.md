## Why

The current ONNX export API has inconsistencies that limit composability and extensibility:
- Only `LinearModel` implements `OnnxExportable` - no abstraction for other model types
- Preprocessing transformers have no export trait - export logic is hardcoded in match statements
- No way to compose custom export logic for new model/transformer types

This makes it impossible for users to extend ONNX export for custom models or transformers without modifying library code.

## What Changes

### New Traits for Composable Export
- `OnnxExportable` trait will be generalized with a `build_onnx_graph()` method
- `OnnxNodeBuilder` trait for preprocessing transformers to contribute nodes to the graph
- Models and transformers can be composed arbitrarily

### **BREAKING** API Changes
- `OnnxExportable::to_onnx()` signature changes to use `OnnxGraphBuilder` context
- Preprocessing export functions move from hardcoded match to trait-based dispatch

### Improvements
- All model types can implement `OnnxExportable` (LinearRegressor, LinearClassifier, etc.)
- Custom preprocessing transformers can be exported by implementing `OnnxNodeBuilder`
- Better validation during graph construction
- Unified error handling across export operations

## Capabilities

### New Capabilities

- `onnx-export-api`: Composable ONNX export API with traits for models and transformers to contribute nodes to the export graph

### Modified Capabilities

- `onnx-pipeline-export`: Updated to use the new composable API instead of hardcoded match statements

## Impact

- **Modified files**: `lib/src/onnx/mod.rs`, `lib/src/onnx/export.rs`, `lib/src/onnx/from_pipeline.rs`
- **New files**: `lib/src/onnx/traits.rs` (new trait definitions)
- **Breaking changes**: `OnnxExportable` trait signature changes
- **Dependencies**: No new external dependencies
