# onnx-deployment-example Specification

## Purpose
TBD - created by archiving change onnx-pipeline-server. Update Purpose after archive.
## Requirements
### Requirement: Complete deployment example
The system SHALL provide a complete, runnable example demonstrating the full ML deployment workflow.

#### Scenario: Example runs successfully
- **WHEN** user runs the example with cargo run --example onnx_deployment
- **THEN** example completes without errors demonstrating all workflow steps

### Requirement: Example demonstrates training
The example SHALL demonstrate model training with preprocessing pipeline.

#### Scenario: Create and train pipeline
- **WHEN** example runs
- **THEN** it creates a preprocessing pipeline, trains a model, and shows training metrics

### Requirement: Example demonstrates export
The example SHALL demonstrate exporting the complete pipeline to ONNX.

#### Scenario: Export pipeline to file
- **WHEN** example runs
- **THEN** it exports the trained pipeline to a .onnx file and verifies the file exists

### Requirement: Example demonstrates inference server
The example SHALL demonstrate starting and using the inference server.

#### Scenario: Start server and make predictions
- **WHEN** example runs
- **THEN** it starts the inference server, makes HTTP prediction requests, and displays results

### Requirement: Example includes comparison
The example SHALL compare native Rust predictions with ONNX predictions.

#### Scenario: Verify prediction parity
- **WHEN** example runs
- **THEN** it compares predictions from native Rust model vs ONNX model and shows they match within tolerance

### Requirement: Example is documented
The example SHALL include comprehensive comments explaining each step.

#### Scenario: Code is well-commented
- **WHEN** user reads the example code
- **THEN** each major step has comments explaining what it does and why

