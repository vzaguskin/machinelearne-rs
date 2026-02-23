//! ONNX inference server for model deployment.
//!
//! This module provides a lightweight HTTP server for deploying ONNX models
//! and serving predictions via REST API.
//!
//! # Features
//!
//! - Single and batch prediction endpoints
//! - Health and readiness checks for orchestration
//! - Configurable host and port
//! - CPU and CUDA execution provider support

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

use super::error::OnnxError;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Host address to bind to.
    pub host: String,
    /// Port to listen on.
    pub port: u16,
    /// Path to the ONNX model file.
    pub model_path: String,
    /// Execution provider to use ("cpu" or "cuda").
    pub provider: ExecutionProvider,
    /// Maximum batch size for predictions.
    pub max_batch_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            model_path: "model.onnx".to_string(),
            provider: ExecutionProvider::Cpu,
            max_batch_size: 1000,
        }
    }
}

/// Execution provider for ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU execution provider.
    Cpu,
    /// CUDA execution provider (requires CUDA support).
    Cuda,
}

impl Default for ExecutionProvider {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Shared server state.
pub struct ServerState {
    /// ONNX inference session.
    session: OnceCell<Arc<Mutex<Session>>>,
    /// Server configuration.
    config: ServerConfig,
    /// Whether the model is loaded and ready.
    ready: std::sync::atomic::AtomicBool,
}

impl ServerState {
    /// Create new server state.
    pub fn new(config: ServerConfig) -> Self {
        Self {
            session: OnceCell::new(),
            config,
            ready: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Get or initialize the inference session.
    pub async fn get_session(&self) -> Result<Arc<Mutex<Session>>, OnnxError> {
        self.session
            .get_or_try_init(|| async {
                let session = self.build_session()?;
                self.ready.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(Arc::new(Mutex::new(session)))
            })
            .await
            .map(Arc::clone)
    }

    fn build_session(&self) -> Result<Session, OnnxError> {
        let builder = Session::builder()
            .map_err(|e| {
                OnnxError::InferenceError(format!("Failed to create session builder: {}", e))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                OnnxError::InferenceError(format!("Failed to set optimization level: {}", e))
            })?;

        let builder = match self.config.provider {
            ExecutionProvider::Cpu => builder,
            ExecutionProvider::Cuda => {
                #[cfg(feature = "onnx-cuda")]
                {
                    builder
                        .with_execution_providers([ort::ep::CUDA::default().build()])
                        .map_err(|e| {
                            OnnxError::InferenceError(format!("Failed to configure CUDA: {}", e))
                        })?
                }
                #[cfg(not(feature = "onnx-cuda"))]
                {
                    // Fall back to CPU if CUDA is not compiled in
                    eprintln!("Warning: CUDA requested but not compiled in, falling back to CPU");
                    builder
                }
            }
        };

        let session = builder
            .commit_from_file(&self.config.model_path)
            .map_err(|e| OnnxError::InferenceError(format!("Failed to load model: {}", e)))?;

        Ok(session)
    }

    /// Check if the server is ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// ============================================================================
// Request/Response types
// ============================================================================

/// Single prediction request.
#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    /// Input features as a flat array.
    pub features: Vec<f32>,
    /// Shape of the input (optional, defaults to 1D).
    pub shape: Option<Vec<usize>>,
}

/// Batch prediction request.
#[derive(Debug, Deserialize)]
pub struct BatchPredictRequest {
    /// Multiple input samples.
    pub samples: Vec<PredictRequest>,
}

/// Single prediction response.
#[derive(Debug, Serialize)]
pub struct PredictResponse {
    /// Prediction output.
    pub prediction: Vec<f32>,
    /// Shape of the output.
    pub shape: Vec<usize>,
}

/// Batch prediction response.
#[derive(Debug, Serialize)]
pub struct BatchPredictResponse {
    /// Predictions for all samples.
    pub predictions: Vec<PredictResponse>,
}

/// Error response.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error message.
    pub error: String,
}

// ============================================================================
// HTTP handlers
// ============================================================================

/// Health check endpoint.
pub async fn health() -> impl IntoResponse {
    StatusCode::OK
}

/// Readiness check endpoint.
pub async fn ready(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    if state.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Single prediction endpoint.
pub async fn predict(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, ServerError> {
    let session = state.get_session().await?;

    // Validate input
    if request.features.is_empty() {
        return Err(ServerError::BadRequest("Empty features".to_string()));
    }

    if request.features.len() > state.config.max_batch_size * 1000 {
        return Err(ServerError::BadRequest(format!(
            "Input too large: {} features exceed limit",
            request.features.len()
        )));
    }

    // Determine input shape
    let shape: Vec<usize> = request
        .shape
        .unwrap_or_else(|| vec![1, request.features.len()]);
    let total_elements: usize = shape.iter().product();

    if total_elements != request.features.len() {
        return Err(ServerError::BadRequest(format!(
            "Shape mismatch: expected {} elements, got {}",
            total_elements,
            request.features.len()
        )));
    }

    // Create input array
    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let data = request.features.into_boxed_slice();

    // Create input value from array
    let input_value = ort::value::Value::from_array((shape_i64, data))
        .map_err(|e| ServerError::Inference(format!("Failed to create input tensor: {}", e)))?;

    // Lock the session and run inference
    let mut session_guard = session
        .lock()
        .map_err(|_| ServerError::Internal("Failed to lock session".to_string()))?;

    let outputs = session_guard
        .run(ort::inputs![input_value])
        .map_err(|e| ServerError::Inference(e.to_string()))?;

    // Extract output - try_extract_tensor returns (&Shape, &[f32])
    let (output_shape, output_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| ServerError::Inference(format!("Failed to extract output: {}", e)))?;

    let output_shape: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();
    let prediction: Vec<f32> = output_data.to_vec();

    Ok(Json(PredictResponse {
        prediction,
        shape: output_shape,
    }))
}

/// Batch prediction endpoint.
pub async fn predict_batch(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<BatchPredictRequest>,
) -> Result<Json<BatchPredictResponse>, ServerError> {
    if request.samples.is_empty() {
        return Err(ServerError::BadRequest("Empty batch".to_string()));
    }

    if request.samples.len() > state.config.max_batch_size {
        return Err(ServerError::BadRequest(format!(
            "Batch size {} exceeds limit {}",
            request.samples.len(),
            state.config.max_batch_size
        )));
    }

    let mut predictions = Vec::with_capacity(request.samples.len());

    for sample in request.samples {
        let single_request = PredictRequest {
            features: sample.features,
            shape: sample.shape,
        };

        let response = predict(State(state.clone()), Json(single_request)).await?;
        predictions.push(response.0);
    }

    Ok(Json(BatchPredictResponse { predictions }))
}

// ============================================================================
// Error handling
// ============================================================================

/// Server error type.
#[derive(Debug)]
pub enum ServerError {
    /// Bad request error.
    BadRequest(String),
    /// Inference error.
    Inference(String),
    /// Internal server error.
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ServerError::Inference(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(ErrorResponse { error: message });
        (status, body).into_response()
    }
}

impl From<OnnxError> for ServerError {
    fn from(err: OnnxError) -> Self {
        ServerError::Internal(err.to_string())
    }
}

// ============================================================================
// Server builder
// ============================================================================

/// ONNX inference server builder.
pub struct OnnxServer {
    config: ServerConfig,
}

impl OnnxServer {
    /// Create a new server builder.
    pub fn new() -> Self {
        Self {
            config: ServerConfig::default(),
        }
    }

    /// Set the host address.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self
    }

    /// Set the port.
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Set the model path.
    pub fn model(mut self, path: impl Into<String>) -> Self {
        self.config.model_path = path.into();
        self
    }

    /// Set the execution provider.
    pub fn provider(mut self, provider: ExecutionProvider) -> Self {
        self.config.provider = provider;
        self
    }

    /// Set the maximum batch size.
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Build the server router.
    pub fn build(self) -> Router {
        let state = Arc::new(ServerState::new(self.config));

        Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .route("/predict", post(predict))
            .route("/predict/batch", post(predict_batch))
            .with_state(state)
    }

    /// Build and run the server.
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        let router = self.build();

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, router).await?;

        Ok(())
    }
}

impl Default for OnnxServer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.provider, ExecutionProvider::Cpu);
        assert_eq!(config.max_batch_size, 1000);
    }

    #[test]
    fn test_server_builder() {
        let server = OnnxServer::new()
            .host("127.0.0.1")
            .port(3000)
            .model("test.onnx")
            .provider(ExecutionProvider::Cpu)
            .max_batch_size(100);

        assert_eq!(server.config.host, "127.0.0.1");
        assert_eq!(server.config.port, 3000);
        assert_eq!(server.config.model_path, "test.onnx");
        assert_eq!(server.config.provider, ExecutionProvider::Cpu);
        assert_eq!(server.config.max_batch_size, 100);
    }

    #[test]
    fn test_predict_request_deserialize() {
        let json = r#"{"features": [1.0, 2.0, 3.0], "shape": [1, 3]}"#;
        let request: PredictRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.features, vec![1.0, 2.0, 3.0]);
        assert_eq!(request.shape, Some(vec![1, 3]));
    }

    #[test]
    fn test_predict_request_without_shape() {
        let json = r#"{"features": [1.0, 2.0, 3.0]}"#;
        let request: PredictRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.features, vec![1.0, 2.0, 3.0]);
        assert!(request.shape.is_none());
    }

    #[test]
    fn test_batch_predict_request_deserialize() {
        let json = r#"{"samples": [{"features": [1.0, 2.0]}, {"features": [3.0, 4.0]}]}"#;
        let request: BatchPredictRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.samples.len(), 2);
    }

    #[test]
    fn test_error_response_serialize() {
        let error = ErrorResponse {
            error: "test error".to_string(),
        };
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("test error"));
    }
}
