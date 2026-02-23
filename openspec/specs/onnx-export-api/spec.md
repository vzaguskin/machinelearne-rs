# onnx-export-api Specification

## Purpose
TBD - created by archiving change consistent-onnx-export-api. Update Purpose after archive.
## Requirements
### Requirement: OnnxNodeBuilder trait for graph contributions
The system SHALL provide an `OnnxNodeBuilder` trait that allows types to contribute nodes to an ONNX graph.

#### Scenario: Transformer implements OnnxNodeBuilder
- **WHEN** a preprocessing transformer implements `OnnxNodeBuilder`
- **THEN** it can add nodes to the graph and return an output tensor name

#### Scenario: OnnxNodeBuilder receives input tensor name
- **WHEN** `build_onnx_nodes` is called on a transformer
- **THEN** it receives the input tensor name to connect its nodes to

### Requirement: OnnxExportable trait for model export
The system SHALL provide an `OnnxExportable` trait with a `build_onnx_graph` method for complete model export.

#### Scenario: Model implements OnnxExportable
- **WHEN** a model implements `OnnxExportable`
- **THEN** it can build a complete ONNX graph using the provided builder

#### Scenario: Default convenience methods
- **WHEN** a type implements `build_onnx_graph`
- **THEN** it automatically gets `to_onnx()` and `save_onnx()` methods with default implementations

### Requirement: Composable export architecture
The system SHALL allow arbitrary composition of `OnnxNodeBuilder` implementations to build complex graphs.

#### Scenario: Chain multiple transformers
- **WHEN** multiple transformers implementing `OnnxNodeBuilder` are chained
- **THEN** each transformer's output becomes the next transformer's input

#### Scenario: Mix library and custom transformers
- **WHEN** a pipeline contains both library-provided and custom transformers
- **THEN** both contribute to the same ONNX graph using the same trait interface

### Requirement: Graph builder enhancements
The system SHALL provide `OnnxGraphBuilder` methods for common graph construction tasks.

#### Scenario: Unique name generation
- **WHEN** `unique_name("node")` is called multiple times
- **THEN** each call returns a unique name like "node", "node_1", "node_2"

#### Scenario: Add initializer tensors
- **WHEN** `add_initializer()` is called with tensor data
- **THEN** the tensor is added to the graph as a constant initializer

#### Scenario: Add operation nodes
- **WHEN** `add_node()` is called with a NodeProto
- **THEN** the node is added to the graph with proper connectivity

### Requirement: Extensibility for custom types
The system SHALL allow users to implement ONNX export for their own models and transformers without modifying library code.

#### Scenario: Custom model export
- **WHEN** a user creates a custom model type
- **THEN** they can implement `OnnxExportable` to enable ONNX export

#### Scenario: Custom transformer export
- **WHEN** a user creates a custom preprocessing transformer
- **THEN** they can implement `OnnxNodeBuilder` to enable pipeline export

