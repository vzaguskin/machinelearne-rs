# onnx-pipeline-export Specification

## Purpose
TBD - created by archiving change onnx-pipeline-server. Update Purpose after archive.
## Requirements
### Requirement: Export complete pipeline to ONNX
The system SHALL allow exporting a complete ML pipeline (preprocessing + model) as a single ONNX file for end-to-end inference.

#### Scenario: Export pipeline with scalers and linear model
- **WHEN** user has a fitted pipeline with StandardScaler and LinearRegression
- **THEN** system exports a single ONNX file that performs both preprocessing and prediction

#### Scenario: Export pipeline with multiple preprocessing steps
- **WHEN** user has a pipeline with SimpleImputer → StandardScaler → PolynomialFeatures → LinearRegression
- **THEN** system exports a single ONNX file with all transformations chained correctly

### Requirement: Pipeline export preserves transformation order
The system SHALL preserve the exact order of preprocessing steps in the exported ONNX graph.

#### Scenario: Verify step order in exported model
- **WHEN** pipeline with steps A → B → C is exported
- **THEN** the ONNX graph executes steps in order A → B → C

### Requirement: Support all preprocessing transformers
The system SHALL support exporting all fitted preprocessing transformers to ONNX format.

#### Scenario: Export StandardScaler
- **WHEN** a fitted StandardScaler is exported
- **THEN** the ONNX graph contains Sub and Div operations with mean and std initializers

#### Scenario: Export MinMaxScaler
- **WHEN** a fitted MinMaxScaler is exported
- **THEN** the ONNX graph contains operations for (X - min) * scale + feature_range_min

#### Scenario: Export SimpleImputer
- **WHEN** a fitted SimpleImputer is exported
- **THEN** the ONNX graph replaces NaN/missing values with learned statistics

### Requirement: Pipeline export validation
The system SHALL validate that the pipeline can be exported before attempting serialization.

#### Scenario: Detect unsupported transformer
- **WHEN** user attempts to export a pipeline with an unsupported transformer
- **THEN** system returns a descriptive error indicating which transformer is not supported

### Requirement: Export with metadata
The system SHALL include metadata in the exported ONNX file for model identification.

#### Scenario: Include model metadata
- **WHEN** pipeline is exported
- **THEN** ONNX file includes producer name, model version, and description

