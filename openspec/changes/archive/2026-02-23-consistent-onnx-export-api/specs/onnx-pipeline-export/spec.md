## MODIFIED Requirements

### Requirement: Support all preprocessing transformers
The system SHALL support exporting all fitted preprocessing transformers to ONNX format via the `OnnxNodeBuilder` trait.

#### Scenario: Export StandardScaler
- **WHEN** a fitted StandardScaler is exported
- **THEN** the ONNX graph contains Sub and Div operations with mean and std initializers

#### Scenario: Export MinMaxScaler
- **WHEN** a fitted MinMaxScaler is exported
- **THEN** the ONNX graph contains operations for (X - min) * scale + feature_range_min

#### Scenario: Export SimpleImputer
- **WHEN** a fitted SimpleImputer is exported
- **THEN** the ONNX graph replaces NaN/missing values with learned statistics

#### Scenario: Custom transformer with OnnxNodeBuilder
- **WHEN** a custom transformer implements `OnnxNodeBuilder`
- **THEN** it can be included in pipeline export without modifying library code

### Requirement: Pipeline export validation
The system SHALL validate that the pipeline can be exported by checking `OnnxNodeBuilder` implementations.

#### Scenario: Detect unsupported transformer
- **WHEN** user attempts to export a pipeline with a transformer not implementing `OnnxNodeBuilder`
- **THEN** system returns a descriptive error indicating which transformer lacks export support

## ADDED Requirements

### Requirement: Trait-based pipeline export
The system SHALL use trait-based dispatch to compose pipeline export from individual `OnnxNodeBuilder` implementations.

#### Scenario: Pipeline uses trait dispatch
- **WHEN** a FittedPipeline is exported
- **THEN** each preprocessing step's `build_onnx_nodes` method is called in sequence

#### Scenario: Output tensor chaining
- **WHEN** multiple preprocessing steps are in a pipeline
- **THEN** each step's output tensor name becomes the next step's input
