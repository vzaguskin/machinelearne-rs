//! OnnxExportable implementations for models.
//!
//! This module provides ONNX export capability for trained models
//! by implementing the [`OnnxExportable`] trait.

use super::error::OnnxError;
use super::graph::OnnxGraphBuilder;
use super::traits::{OnnxExportable, OnnxNodeBuilder};
use crate::backend::Backend;
use crate::model::linear::{Fitted, LinearModel};
use crate::model::mlp::MLPModel;
use crate::model::InferenceModel;
use crate::pipeline::FittedPipeline;

impl<B: Backend> OnnxExportable for LinearModel<B, Fitted> {
    fn build_onnx_graph(&self, builder: &mut OnnxGraphBuilder) -> Result<String, OnnxError> {
        let params = self.extract_params();
        let n_features = params.weights.len();

        // Add input: [batch_size, n_features]
        builder.add_input_float("input", n_features);

        // Get weights and bias
        let weights: Vec<f32> = params.weights.iter().map(|&w| w as f32).collect();
        let bias: f32 = params.bias;

        // Add weights as initializer: shape [1, n_features] for Gemm
        // We want output = input @ weights^T + bias
        // Gemm: Y = alpha * A @ B + beta * C
        builder.add_float_initializer("weights", &[1, n_features as i64], &weights);
        builder.add_float_initializer("bias", &[1], &[bias]);

        // Add Gemm node: output = input @ weights^T + bias
        builder.gemm(
            "input",
            "weights",
            Some("bias"),
            false, // transA
            true,  // transB - transpose weights from [1, F] to [F, 1]
            1.0,   // alpha
            1.0,   // beta
            "output",
        );

        // Add output
        builder.add_output_float("output", 1);

        Ok("output".to_string())
    }
}

// =============================================================================
// MLP Model
// =============================================================================

impl<B: Backend> OnnxExportable for MLPModel<B, Fitted> {
    fn build_onnx_graph(&self, builder: &mut OnnxGraphBuilder) -> Result<String, OnnxError> {
        let layer_sizes = self.layer_sizes();
        let n_features_in = layer_sizes.first().copied().unwrap_or(0);
        let n_features_out = layer_sizes.last().copied().unwrap_or(0);

        // Add input: [batch_size, n_features_in]
        builder.add_input_float("input", n_features_in);

        // Track current tensor name
        let mut current = "input".to_string();

        // Get layers and activations
        let layers = self.layers();
        let activations = self.activations();

        // Export each layer
        for (layer_idx, layer) in layers.iter().enumerate() {
            let (out_features, in_features) = layer.weights.shape();

            // Get weights and bias as f32 vectors
            let weights: Vec<f32> = layer
                .weights
                .ravel()
                .to_vec()
                .into_iter()
                .map(|x| x as f32)
                .collect();
            let bias: Vec<f32> = layer.bias.to_vec().into_iter().map(|x| x as f32).collect();

            // Create unique names for this layer's weights and bias
            let weights_name = format!("layer{}_weights", layer_idx);
            let bias_name = format!("layer{}_bias", layer_idx);
            let gemm_output = format!("layer{}_gemm", layer_idx);
            let activation_output = format!("layer{}_act", layer_idx);

            // Add weights as initializer: shape [out_features, in_features]
            builder.add_float_initializer(
                &weights_name,
                &[out_features as i64, in_features as i64],
                &weights,
            );

            // Add bias as initializer: shape [out_features]
            builder.add_float_initializer(&bias_name, &[out_features as i64], &bias);

            // Add Gemm node: output = input @ weights^T + bias
            // For MLP: Y = X @ W^T + b where W is [out, in]
            builder.gemm(
                &current,
                &weights_name,
                Some(&bias_name),
                false, // transA
                true,  // transB - transpose weights
                1.0,   // alpha
                1.0,   // beta
                &gemm_output,
            );

            // Add activation node
            let activation = activations[layer_idx];
            match activation {
                crate::model::Activation::ReLU => {
                    builder.relu(&gemm_output, &activation_output);
                }
                crate::model::Activation::Sigmoid => {
                    builder.sigmoid(&gemm_output, &activation_output);
                }
                crate::model::Activation::Tanh => {
                    builder.tanh(&gemm_output, &activation_output);
                }
                crate::model::Activation::Identity => {
                    // Identity: just use the gemm output directly
                    current = gemm_output;
                    continue;
                }
            }

            current = activation_output;
        }

        // Add output
        builder.add_output_float(&current, n_features_out);

        Ok(current)
    }
}

// =============================================================================
// FittedPipeline
// =============================================================================

impl<B: Backend> OnnxExportable for FittedPipeline<B> {
    fn build_onnx_graph(&self, builder: &mut OnnxGraphBuilder) -> Result<String, OnnxError> {
        // Add input with original feature count
        let n_features_in = self.n_features_in();
        builder.add_input_float("input", n_features_in);

        // Track current tensor name
        let mut current = "input".to_string();

        // Export preprocessing steps using trait dispatch
        if let Some(preproc) = self.preprocessor() {
            for step in preproc.steps() {
                current = step.build_onnx_nodes(builder, &current)?;
            }
        }

        // Export polynomial features if present
        if let Some(poly) = self.polynomial() {
            current = poly.build_onnx_nodes(builder, &current)?;
        }

        // Export the linear model
        let model = self.model();
        let params = model.extract_params();
        let n_features = params.weights.len();

        // Add model weights
        let weights: Vec<f32> = params.weights.iter().map(|&w| w as f32).collect();
        let bias: f32 = params.bias;

        builder.add_float_initializer("model_weights", &[1, n_features as i64], &weights);
        builder.add_float_initializer("model_bias", &[1], &[bias]);

        // Add Gemm for linear model
        builder.gemm(
            &current,
            "model_weights",
            Some("model_bias"),
            false,
            true,
            1.0,
            1.0,
            "output",
        );

        // Add output
        builder.add_output_float("output", 1);

        Ok("output".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CpuBackend, Scalar, Tensor1D};
    use crate::model::linear::LinearParams;

    fn create_test_model() -> LinearModel<CpuBackend, Fitted> {
        let params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0, 3.0]),
            bias: Scalar::new(0.5),
        };
        LinearModel::<CpuBackend, Fitted>::new(params)
    }

    #[test]
    fn test_linear_model_build_onnx_graph() {
        let model = create_test_model();
        let mut builder = OnnxGraphBuilder::new("linear_model");
        builder.set_metadata("machinelearne-rs", env!("CARGO_PKG_VERSION"), None);

        let output = model.build_onnx_graph(&mut builder).unwrap();
        assert_eq!(output, "output");
        assert!(!builder.graph.initializer.is_empty());
        assert!(!builder.graph.node.is_empty());
    }

    #[test]
    fn test_linear_model_to_onnx() {
        let model = create_test_model();
        let bytes = model.to_onnx("test_model").unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_linear_model_to_onnx_default() {
        let model = create_test_model();
        let bytes = model.to_onnx_default().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_linear_model_save_onnx() {
        let model = create_test_model();

        let temp_file = std::env::temp_dir().join("test_linear_model_new.onnx");
        model.save_onnx(&temp_file, Some("linear_model")).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_linear_model_save_onnx_to_path() {
        let model = create_test_model();

        let temp_file = std::env::temp_dir().join("my_linear_model.onnx");
        model.save_onnx_to_path(&temp_file).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_linear_model_metadata_in_output() {
        let model = create_test_model();
        let bytes = model.to_onnx("test_model").unwrap();

        // The metadata should be included in the serialized model
        let bytes_str = String::from_utf8_lossy(&bytes);
        assert!(bytes_str.contains("machinelearne-rs"));
    }

    // ========================================================================
    // Pipeline Tests
    // ========================================================================

    fn create_test_pipeline() -> FittedPipeline<CpuBackend> {
        use crate::backend::Tensor2D;
        use crate::model::linear::LinearRegression;
        use crate::model::TrainableModel;
        use crate::preprocessing::pipeline::Pipeline;
        use crate::preprocessing::scaling::StandardScaler;
        use crate::preprocessing::traits::Transformer;

        // Create preprocessing pipeline
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let preproc = Pipeline::<CpuBackend>::new().add_standard_scaler(StandardScaler::new());
        let fitted_preproc = preproc.fit(&data).unwrap();

        // Create model
        let model_params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![2.0, 3.0]),
            bias: Scalar::new(1.0),
        };
        let model = LinearRegression::<CpuBackend>::from_params(model_params).into_fitted();

        FittedPipeline::new(Some(fitted_preproc), None, model)
    }

    #[test]
    fn test_pipeline_build_onnx_graph() {
        let pipeline = create_test_pipeline();
        let mut builder = OnnxGraphBuilder::new("pipeline");
        builder.set_metadata("machinelearne-rs", env!("CARGO_PKG_VERSION"), None);

        let output = pipeline.build_onnx_graph(&mut builder).unwrap();
        assert_eq!(output, "output");
        assert!(!builder.graph.initializer.is_empty());
        assert!(!builder.graph.node.is_empty());
    }

    #[test]
    fn test_pipeline_to_onnx() {
        let pipeline = create_test_pipeline();
        let bytes = pipeline.to_onnx("test_pipeline").unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_pipeline_save_onnx() {
        let pipeline = create_test_pipeline();

        let temp_file = std::env::temp_dir().join("test_pipeline.onnx");
        pipeline.save_onnx(&temp_file, Some("pipeline")).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_pipeline_save_onnx_to_path() {
        let pipeline = create_test_pipeline();

        let temp_file = std::env::temp_dir().join("my_pipeline.onnx");
        pipeline.save_onnx_to_path(&temp_file).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_pipeline_model_only() {
        // Pipeline with just a model (no preprocessing)
        let model_params = LinearParams {
            weights: Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]),
            bias: Scalar::new(0.5),
        };
        let model = LinearModel::<CpuBackend, Fitted>::new(model_params);
        let pipeline = FittedPipeline::from_model(model);

        let bytes = pipeline.to_onnx("model_only_pipeline").unwrap();
        assert!(!bytes.is_empty());
    }

    // ========================================================================
    // MLP Model Tests
    // ========================================================================

    fn create_test_mlp() -> MLPModel<CpuBackend, Fitted> {
        use crate::model::{Activation, TrainableModel, MLP};

        // Create: 2 inputs -> 4 hidden (ReLU) -> 1 output (Identity)
        let model = MLP::<CpuBackend>::new(&[2, 4, 1], &[Activation::ReLU, Activation::Identity]);
        model.into_fitted()
    }

    #[test]
    fn test_mlp_build_onnx_graph() {
        let model = create_test_mlp();
        let mut builder = OnnxGraphBuilder::new("mlp_model");
        builder.set_metadata("machinelearne-rs", env!("CARGO_PKG_VERSION"), None);

        let output = model.build_onnx_graph(&mut builder).unwrap();
        assert!(!output.is_empty());
        assert!(!builder.graph.initializer.is_empty());
        assert!(!builder.graph.node.is_empty());
    }

    #[test]
    fn test_mlp_to_onnx() {
        let model = create_test_mlp();
        let bytes = model.to_onnx("test_mlp").unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_mlp_to_onnx_default() {
        let model = create_test_mlp();
        let bytes = model.to_onnx_default().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_mlp_save_onnx() {
        let model = create_test_mlp();

        let temp_file = std::env::temp_dir().join("test_mlp_model.onnx");
        model.save_onnx(&temp_file, Some("mlp_model")).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_mlp_save_onnx_to_path() {
        let model = create_test_mlp();

        let temp_file = std::env::temp_dir().join("my_mlp_model.onnx");
        model.save_onnx_to_path(&temp_file).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_mlp_with_various_activations() {
        use crate::model::{Activation, TrainableModel, MLP};

        // Test with different activation combinations
        let activations = vec![
            &[Activation::Sigmoid, Activation::Identity][..],
            &[Activation::Tanh, Activation::Identity][..],
            &[Activation::ReLU, Activation::Sigmoid][..],
        ];

        for acts in activations {
            let model = MLP::<CpuBackend>::new(&[2, 4, 1], acts);
            let fitted = model.into_fitted();
            let bytes = fitted.to_onnx("test_mlp_activations").unwrap();
            assert!(!bytes.is_empty());
        }
    }
}
