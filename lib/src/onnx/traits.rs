//! Trait definitions for ONNX export.
//!
//! This module provides the core traits for composable ONNX export:
//! - [`OnnxNodeBuilder`] for types that contribute nodes to an ONNX graph
//! - [`OnnxExportable`] for types that can be exported as complete ONNX models

use super::error::OnnxError;
use super::graph::OnnxGraphBuilder;
use std::path::Path;

/// Trait for types that can contribute nodes to an ONNX graph.
///
/// This trait is implemented by preprocessing transformers and other types
/// that need to add computation nodes to an ONNX graph. The trait enables
/// composable export where multiple transformers can be chained together.
///
/// # Example
///
/// ```rust,ignore
/// use machinelearne_rs::onnx::{OnnxNodeBuilder, OnnxGraphBuilder, OnnxError};
///
/// struct MyTransformer {
///     scale: f32,
/// }
///
/// impl OnnxNodeBuilder for MyTransformer {
///     fn build_onnx_nodes(
///         &self,
///         builder: &mut OnnxGraphBuilder,
///         input_name: &str,
///     ) -> Result<String, OnnxError> {
///         // Add a scale initializer
///         let output_name = builder.unique_name("scaled");
///         builder.add_float_initializer("my_scale", &[1], &[self.scale]);
///
///         // Add Mul node: output = input * scale
///         builder.add_node(
///             "Mul",
///             vec![input_name.to_string(), "my_scale".to_string()],
///             vec![output_name.clone()],
///             vec![],
///         );
///
///         Ok(output_name)
///     }
/// }
/// ```
pub trait OnnxNodeBuilder {
    /// Build ONNX nodes for this transformer.
    ///
    /// Implementations should add nodes to the graph builder and return
    /// the name of the output tensor that subsequent nodes should connect to.
    ///
    /// # Arguments
    /// * `builder` - The graph builder to add nodes to
    /// * `input_name` - The name of the input tensor to connect to
    ///
    /// # Returns
    /// The name of the output tensor for the next step to use.
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError>;
}

/// Trait for types that can be exported as complete ONNX models.
///
/// This trait is the main entry point for ONNX export. Types implementing
/// this trait can build a complete ONNX graph and serialize it.
///
/// The trait provides default implementations for convenience methods
/// like [`to_onnx`](OnnxExportable::to_onnx) and
/// [`save_onnx`](OnnxExportable::save_onnx).
///
/// # Example
///
/// ```rust,ignore
/// use machinelearne_rs::onnx::{OnnxExportable, OnnxGraphBuilder, OnnxError};
///
/// struct MyModel {
///     weights: Vec<f32>,
///     bias: f32,
/// }
///
/// impl OnnxExportable for MyModel {
///     fn build_onnx_graph(
///         &self,
///         builder: &mut OnnxGraphBuilder,
///     ) -> Result<String, OnnxError> {
///         let n_features = self.weights.len();
///
///         // Add input
///         builder.add_input_float("input", n_features);
///
///         // Add weights and bias
///         builder.add_float_initializer("weights", &[n_features as i64], &self.weights);
///         builder.add_float_initializer("bias", &[1], &[self.bias]);
///
///         // Add computation nodes
///         // ... (Gemm, etc.)
///
///         // Add output
///         builder.add_output_float("output", 1);
///
///         Ok("output".to_string())
///     }
/// }
///
/// // Now you can export the model
/// let model = MyModel { weights: vec![1.0, 2.0], bias: 0.5 };
/// let bytes = model.to_onnx()?;  // Uses default implementation
/// model.save_onnx("model.onnx")?;  // Uses default implementation
/// ```
pub trait OnnxExportable {
    /// Build the ONNX graph for this model.
    ///
    /// Implementations should:
    /// 1. Add input tensors via `builder.add_input_float()` or similar
    /// 2. Add initializers for weights/biases
    /// 3. Add computation nodes
    /// 4. Add output tensors
    ///
    /// # Arguments
    /// * `builder` - A fresh graph builder to populate
    ///
    /// # Returns
    /// The name of the final output tensor.
    fn build_onnx_graph(&self, builder: &mut OnnxGraphBuilder) -> Result<String, OnnxError>;

    /// Export the model to ONNX format as bytes with a specific model name.
    ///
    /// This default implementation creates a new graph builder,
    /// calls `build_onnx_graph`, and serializes the result.
    ///
    /// # Arguments
    /// * `model_name` - Name for the ONNX model
    ///
    /// # Returns
    /// Serialized ONNX model bytes.
    fn to_onnx(&self, model_name: &str) -> Result<Vec<u8>, OnnxError> {
        let mut builder = OnnxGraphBuilder::new(model_name);
        builder.set_metadata("machinelearne-rs", env!("CARGO_PKG_VERSION"), None);
        self.build_onnx_graph(&mut builder)?;
        builder.build()
    }

    /// Export the model to ONNX format with a default model name.
    ///
    /// Convenience method for quick exports.
    fn to_onnx_default(&self) -> Result<Vec<u8>, OnnxError> {
        self.to_onnx("model")
    }

    /// Save the model to an ONNX file with an optional model name.
    ///
    /// # Arguments
    /// * `path` - Path to save the ONNX file
    /// * `model_name` - Optional name for the model (defaults to file stem or "model")
    fn save_onnx<P: AsRef<Path>>(
        &self,
        path: P,
        model_name: Option<&str>,
    ) -> Result<(), OnnxError> {
        let name = model_name.map(|s| s.to_string()).unwrap_or_else(|| {
            path.as_ref()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string()
        });
        let bytes = self.to_onnx(&name)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Save the model to an ONNX file, inferring the model name from the file path.
    ///
    /// Convenience method for common use case.
    ///
    /// # Arguments
    /// * `path` - Path to save the ONNX file
    fn save_onnx_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), OnnxError> {
        self.save_onnx(path, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleTransformer {
        scale: f32,
    }

    impl OnnxNodeBuilder for SimpleTransformer {
        fn build_onnx_nodes(
            &self,
            builder: &mut OnnxGraphBuilder,
            input_name: &str,
        ) -> Result<String, OnnxError> {
            let output_name = builder.unique_name("scaled");
            builder.add_float_initializer("scale", &[1], &[self.scale]);
            builder.add_node(
                "Mul",
                vec![input_name.to_string(), "scale".to_string()],
                vec![output_name.clone()],
                vec![],
            );
            Ok(output_name)
        }
    }

    struct SimpleModel {
        weights: Vec<f32>,
        bias: f32,
    }

    impl OnnxExportable for SimpleModel {
        fn build_onnx_graph(&self, builder: &mut OnnxGraphBuilder) -> Result<String, OnnxError> {
            let n_features = self.weights.len();
            builder.add_input_float("input", n_features);
            builder.add_float_initializer("weights", &[n_features as i64], &self.weights);
            builder.add_float_initializer("bias", &[1], &[self.bias]);
            builder.add_output_float("output", 1);

            // Simple matmul + add (simplified for test)
            Ok("output".to_string())
        }
    }

    #[test]
    fn test_onnx_node_builder() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 2);

        let transformer = SimpleTransformer { scale: 2.0 };
        let output = transformer.build_onnx_nodes(&mut builder, "input").unwrap();

        assert!(output.starts_with("scaled"));
        assert!(!builder.graph.initializer.is_empty());
        assert!(!builder.graph.node.is_empty());
    }

    #[test]
    fn test_onnx_exportable() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0],
            bias: 0.5,
        };

        let bytes = model.to_onnx("simple_model").unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_onnx_exportable_default() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0],
            bias: 0.5,
        };

        let bytes = model.to_onnx_default().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_save_onnx() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0],
            bias: 0.5,
        };

        let temp_file = std::env::temp_dir().join("test_traits_model.onnx");
        model.save_onnx(&temp_file, Some("test_model")).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_save_onnx_to_path() {
        let model = SimpleModel {
            weights: vec![1.0, 2.0],
            bias: 0.5,
        };

        let temp_file = std::env::temp_dir().join("my_model.onnx");
        model.save_onnx_to_path(&temp_file).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_chained_transformers() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 2);

        let t1 = SimpleTransformer { scale: 2.0 };
        let t2 = SimpleTransformer { scale: 3.0 };

        let out1 = t1.build_onnx_nodes(&mut builder, "input").unwrap();
        let out2 = t2.build_onnx_nodes(&mut builder, &out1).unwrap();

        assert_ne!(out1, out2);
        assert_eq!(builder.graph.node.len(), 2);
    }
}
