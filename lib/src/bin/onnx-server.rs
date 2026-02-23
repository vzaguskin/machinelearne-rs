//! ONNX Inference Server Binary
//!
//! A standalone HTTP server for serving ONNX model predictions.
//!
//! # Usage
//!
//! ```bash
//! # Start server with default settings
//! cargo run --bin onnx-server --features onnx-server -- --model model.onnx
//!
//! # Start server with custom settings
//! cargo run --bin onnx-server --features onnx-server -- \
//!     --model model.onnx \
//!     --host 127.0.0.1 \
//!     --port 3000 \
//!     --provider cpu
//! ```
//!
//! # Endpoints
//!
//! - `GET /health` - Health check (always returns 200 OK)
//! - `GET /ready` - Readiness check (returns 200 if model loaded, 503 otherwise)
//! - `POST /predict` - Single prediction
//! - `POST /predict/batch` - Batch predictions

#[cfg(feature = "onnx-server")]
use machinelearne_rs::onnx::server::{ExecutionProvider, OnnxServer};

#[cfg(feature = "onnx-server")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;

    /// ONNX Inference Server
    #[derive(Parser, Debug)]
    #[command(name = "onnx-server")]
    #[command(about = "HTTP server for ONNX model inference")]
    struct Args {
        /// Path to the ONNX model file
        #[arg(short, long, default_value = "model.onnx")]
        model: String,

        /// Host address to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Execution provider (cpu or cuda)
        #[arg(long, default_value = "cpu")]
        provider: String,

        /// Maximum batch size for predictions
        #[arg(long, default_value_t = 1000)]
        max_batch_size: usize,
    }

    let args = Args::parse();

    // Parse execution provider
    let provider = match args.provider.to_lowercase().as_str() {
        "cpu" => ExecutionProvider::Cpu,
        "cuda" => ExecutionProvider::Cuda,
        _ => {
            eprintln!("Warning: Unknown provider '{}', using CPU", args.provider);
            ExecutionProvider::Cpu
        }
    };

    println!("Starting ONNX Inference Server...");
    println!("  Model: {}", args.model);
    println!("  Address: {}:{}", args.host, args.port);
    println!("  Provider: {:?}", provider);

    let server = OnnxServer::new()
        .model(&args.model)
        .host(&args.host)
        .port(args.port)
        .provider(provider)
        .max_batch_size(args.max_batch_size);

    println!("Server ready at http://{}:{}/", args.host, args.port);
    println!("Endpoints:");
    println!("  GET  /health        - Health check");
    println!("  GET  /ready         - Readiness check");
    println!("  POST /predict       - Single prediction");
    println!("  POST /predict/batch - Batch predictions");

    server.run().await?;

    Ok(())
}

#[cfg(not(feature = "onnx-server"))]
fn main() {
    eprintln!("Error: This binary requires the 'onnx-server' feature.");
    eprintln!("Please rebuild with: cargo run --bin onnx-server --features onnx-server");
    std::process::exit(1);
}
