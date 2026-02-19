# Spec: ONNX Export

## ADDED

### OnnxExportable Trait

```rust
pub trait OnnxExportable {
    fn to_onnx(&self, opset_version: i64) -> Result<Vec<u8>, OnnxError>;
    fn to_onnx_default(&self) -> Result<Vec<u8>, OnnxError>;
    fn save_onnx<P: AsRef<Path>>(&self, path: P) -> Result<(), OnnxError>;
}
```

### OnnxError

```rust
pub enum OnnxError {
    IoError(io::Error),
    ProtobufError(String),
    ModelCreationError(String),
    InferenceError(String),
    UnsupportedOperation(String),
    InvalidParameters(String),
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    MissingField(String),
}
```

### OnnxGraphBuilder

Builder for constructing ONNX computation graphs.

### OnnxInferenceSession (Experimental)

ONNX Runtime inference wrapper.

## Overview

The ONNX export capability allows trained models and pipelines to be exported to ONNX format for portable deployment.

## Capabilities

### onnx-export

**Description:** Export trained models to ONNX format

**Acceptance Criteria:**
- LinearModel<Fitted> exports to valid ONNX file
- Exported model loads in ONNX Runtime
- Predictions match native Rust implementation

**API:**

```rust
pub trait OnnxExportable {
    fn to_onnx(&self, opset_version: i64) -> Result<Vec<u8>, OnnxError>;
    fn to_onnx_default(&self) -> Result<Vec<u8>, OnnxError>;
    fn save_onnx<P: AsRef<Path>>(&self, path: P) -> Result<(), OnnxError>;
}

// Implemented for:
// - LinearModel<B, Fitted>
// - FittedPipeline<B> (via export_pipeline_to_onnx)
```

**Feature Flag:** `onnx`

### onnx-inference

**Description:** Load and run ONNX models using ONNX Runtime (experimental)

**Acceptance Criteria:**
- Load ONNX file and run batch inference
- Support single-sample prediction

**API:**

```rust
pub struct OnnxInferenceSession { ... }

impl OnnxInferenceSession {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, OnnxError>;
    pub fn predict(&self, input: &Array2<f32>) -> Result<Array1<f32>, OnnxError>;
    pub fn predict_one(&self, input: &Array1<f32>) -> Result<f32, OnnxError>;
}
```

**Feature Flag:** `onnx-inference` (experimental)

## Constraints

- Default opset version: 17
- Supported element types: float32
- Batch dimension is dynamic

## Error Handling

```rust
pub enum OnnxError {
    IoError(io::Error),
    ProtobufError(String),
    ModelCreationError(String),
    InferenceError(String),
    UnsupportedOperation(String),
    InvalidParameters(String),
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    MissingField(String),
}
```
