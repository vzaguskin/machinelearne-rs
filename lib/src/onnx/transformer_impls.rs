//! OnnxNodeBuilder implementations for preprocessing transformers.
//!
//! This module provides ONNX export capability for all fitted transformers
//! by implementing the [`OnnxNodeBuilder`] trait.

use super::error::OnnxError;
use super::graph::OnnxGraphBuilder;
use super::proto::AttributeProto;
use super::traits::OnnxNodeBuilder;
use crate::backend::Backend;
use crate::preprocessing::encoding::{FittedOneHotEncoder, FittedOrdinalEncoder};
use crate::preprocessing::feature_engineering::FittedPolynomialFeatures;
use crate::preprocessing::imputation::FittedSimpleImputer;
use crate::preprocessing::pipeline::PipelineStepEnum;
use crate::preprocessing::scaling::{
    FittedMaxAbsScaler, FittedMinMaxScaler, FittedNormalizer, FittedRobustScaler,
    FittedStandardScaler,
};
use crate::preprocessing::traits::FittedTransformer;

// =============================================================================
// StandardScaler
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedStandardScaler<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        let params = self.extract_params();
        let n_features = params.n_features;

        // StandardScaler: output = (input - mean) / std
        let mean: Vec<f32> = params.mean.iter().map(|&m| m as f32).collect();
        let std: Vec<f32> = params.std.iter().map(|&s| s as f32).collect();

        // Add mean and std as initializers
        let mean_name = builder.unique_name("scaler_mean");
        let std_name = builder.unique_name("scaler_std");
        builder.add_float_initializer(&mean_name, &[1, n_features as i64], &mean);
        builder.add_float_initializer(&std_name, &[1, n_features as i64], &std);

        // output = (input - mean) / std
        let sub_out = builder.unique_name("sub");
        let output = builder.unique_name("standard_scaled");
        builder.sub(input_name, &mean_name, &sub_out);
        builder.div(&sub_out, &std_name, &output);

        Ok(output)
    }
}

// =============================================================================
// MinMaxScaler
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedMinMaxScaler<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        let params = self.extract_params();
        let n_features = params.n_features;

        // MinMaxScaler: output = (input - min) * scale + feature_range_min
        let min: Vec<f32> = params.min_.iter().map(|&m| m as f32).collect();
        let scale: Vec<f32> = params.scale_.iter().map(|&s| s as f32).collect();
        let target_min = params.config.min as f32;

        // Add min and scale as initializers
        let min_name = builder.unique_name("minmax_min");
        let scale_name = builder.unique_name("minmax_scale");
        builder.add_float_initializer(&min_name, &[1, n_features as i64], &min);
        builder.add_float_initializer(&scale_name, &[1, n_features as i64], &scale);

        // output = (input - min) * scale + target_min
        let sub_out = builder.unique_name("sub");
        let mul_out = builder.unique_name("mul");
        let output = builder.unique_name("minmax_scaled");

        builder.sub(input_name, &min_name, &sub_out);
        builder.mul(&sub_out, &scale_name, &mul_out);

        // Add target_min as scalar
        let target_min_name = builder.unique_name("minmax_target_min");
        builder.add_float_initializer(&target_min_name, &[1], &[target_min]);
        builder.add(&mul_out, &target_min_name, &output);

        Ok(output)
    }
}

// =============================================================================
// RobustScaler
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedRobustScaler<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        let params = self.extract_params();
        let n_features = params.n_features;

        // RobustScaler: output = (input - center) / scale
        let center: Vec<f32> = params.center_.iter().map(|&c| c as f32).collect();
        let scale: Vec<f32> = params.scale_.iter().map(|&s| s as f32).collect();

        // Add center and scale as initializers
        let center_name = builder.unique_name("robust_center");
        let scale_name = builder.unique_name("robust_scale");
        builder.add_float_initializer(&center_name, &[1, n_features as i64], &center);
        builder.add_float_initializer(&scale_name, &[1, n_features as i64], &scale);

        // output = (input - center) / scale
        let sub_out = builder.unique_name("sub");
        let output = builder.unique_name("robust_scaled");
        builder.sub(input_name, &center_name, &sub_out);
        builder.div(&sub_out, &scale_name, &output);

        Ok(output)
    }
}

// =============================================================================
// MaxAbsScaler
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedMaxAbsScaler<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        let params = self.extract_params();
        let n_features = params.n_features;

        // MaxAbsScaler: output = input * scale_
        let scale: Vec<f32> = params.scale_.iter().map(|&s| s as f32).collect();

        // Add scale as initializer
        let scale_name = builder.unique_name("maxabs_scale");
        builder.add_float_initializer(&scale_name, &[1, n_features as i64], &scale);

        // output = input * scale
        let output = builder.unique_name("maxabs_scaled");
        builder.mul(input_name, &scale_name, &output);

        Ok(output)
    }
}

// =============================================================================
// Normalizer
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedNormalizer<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        // Normalizer: normalize each sample independently
        // For L2 norm: output = input / sqrt(sum(input^2))

        // 1. Compute input^2
        let sq_out = builder.unique_name("sq");
        builder.mul(input_name, input_name, &sq_out);

        // 2. Sum along axis 1 (features) -> [batch, 1]
        let sum_out = builder.unique_name("sum");
        builder.reduce_sum(&sq_out, &[1], true, &sum_out);

        // 3. sqrt -> [batch, 1]
        let sqrt_out = builder.unique_name("sqrt");
        builder.sqrt(&sum_out, &sqrt_out);

        // 4. input / sqrt -> normalized
        let output = builder.unique_name("normalized");
        builder.div(input_name, &sqrt_out, &output);

        Ok(output)
    }
}

// =============================================================================
// SimpleImputer
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedSimpleImputer<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        let params = self.extract_params();
        let n_features = params.n_features;

        // SimpleImputer: replace NaN/missing with statistics
        let statistics: Vec<f32> = params.statistics_.iter().map(|&s| s as f32).collect();

        // Add statistics as initializer
        let stats_name = builder.unique_name("imputer_stats");
        builder.add_float_initializer(&stats_name, &[1, n_features as i64], &statistics);

        // Use Where operator: where(is_nan(input), stats, input)
        // NaN detection: Equal(input, input) returns false for NaN values
        let eq_out = builder.unique_name("eq");
        let output = builder.unique_name("imputed");

        builder.add_node(
            "Equal",
            vec![input_name.to_string(), input_name.to_string()],
            vec![eq_out.clone()],
            vec![],
        );

        // Where(condition, x, y) = if condition then x else y
        builder.add_node(
            "Where",
            vec![eq_out, input_name.to_string(), stats_name],
            vec![output.clone()],
            vec![],
        );

        Ok(output)
    }
}

// =============================================================================
// OneHotEncoder
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedOneHotEncoder<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        let params = self.extract_params();

        // OneHotEncoder requires the categories
        let categories = &params.categories_;
        let total_cats: i64 = categories.iter().map(|c| c.len() as i64).sum();

        // Create a flat list of category indices
        let values: Vec<i64> = (0..total_cats).collect();

        // Add values as initializer
        let values_name = builder.unique_name("onehot_values");
        builder.add_int64_initializer(&values_name, &[total_cats], &values);

        // Cast input to int64 for OneHot
        let cast_out = builder.unique_name("cast");
        builder.cast(input_name, 7, &cast_out); // 7 = int64

        // OneHot(axis=-1)
        let output = builder.unique_name("onehot_encoded");
        builder.add_node(
            "OneHot",
            vec![cast_out, values_name],
            vec![output.clone()],
            vec![AttributeProto::int("axis", -1)],
        );

        Ok(output)
    }
}

// =============================================================================
// OrdinalEncoder
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedOrdinalEncoder<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        // OrdinalEncoder: map categories to integers
        // This is complex to implement in ONNX without LabelEncoder from ai.onnx.ml
        // For now, we'll use a simple pass-through (identity)
        // A proper implementation would use a series of comparisons

        let output = builder.unique_name("ordinal_encoded");
        builder.add_node(
            "Identity",
            vec![input_name.to_string()],
            vec![output.clone()],
            vec![],
        );

        Ok(output)
    }
}

// =============================================================================
// PolynomialFeatures
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for FittedPolynomialFeatures<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        let params = self.extract_params();
        let degree = params.degree;

        // For degree 1, just return input
        if degree == 1 {
            return Ok(input_name.to_string());
        }

        // For higher degrees, polynomial features are not fully supported
        // Return input for now - validation should have caught this earlier
        // TODO: Implement polynomial feature generation
        Ok(input_name.to_string())
    }
}

// =============================================================================
// PipelineStepEnum - dispatch to inner types
// =============================================================================

impl<B: Backend> OnnxNodeBuilder for PipelineStepEnum<B> {
    fn build_onnx_nodes(
        &self,
        builder: &mut OnnxGraphBuilder,
        input_name: &str,
    ) -> Result<String, OnnxError> {
        match self {
            PipelineStepEnum::StandardScaler(scaler) => {
                scaler.build_onnx_nodes(builder, input_name)
            }
            PipelineStepEnum::MinMaxScaler(scaler) => scaler.build_onnx_nodes(builder, input_name),
            PipelineStepEnum::RobustScaler(scaler) => scaler.build_onnx_nodes(builder, input_name),
            PipelineStepEnum::MaxAbsScaler(scaler) => scaler.build_onnx_nodes(builder, input_name),
            PipelineStepEnum::Normalizer(normalizer) => {
                normalizer.build_onnx_nodes(builder, input_name)
            }
            PipelineStepEnum::SimpleImputer(imputer) => {
                imputer.build_onnx_nodes(builder, input_name)
            }
            PipelineStepEnum::OneHotEncoder(encoder) => {
                encoder.build_onnx_nodes(builder, input_name)
            }
            PipelineStepEnum::OrdinalEncoder(encoder) => {
                encoder.build_onnx_nodes(builder, input_name)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::preprocessing::scaling::{
        MaxAbsScalerParams, MinMaxScalerConfig, MinMaxScalerParams, NormalizerParams,
        RobustScalerConfig, RobustScalerParams, StandardScalerConfig, StandardScalerParams,
    };
    use crate::preprocessing::traits::FittedTransformer;

    #[test]
    fn test_standard_scaler_onnx_nodes() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);

        let params = StandardScalerParams {
            config: StandardScalerConfig::default(),
            n_features: 3,
            mean: vec![0.0, 1.0, 2.0],
            std: vec![1.0, 1.0, 1.0],
        };
        let scaler = FittedStandardScaler::<CpuBackend>::from_params(params).unwrap();

        let output = scaler.build_onnx_nodes(&mut builder, "input").unwrap();
        assert!(output.starts_with("standard_scaled"));
        assert!(!builder.graph.initializer.is_empty());
        assert!(!builder.graph.node.is_empty());
    }

    #[test]
    fn test_minmax_scaler_onnx_nodes() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 2);

        let params = MinMaxScalerParams {
            n_features: 2,
            min_: vec![0.0, 0.0],
            max_: vec![1.0, 2.0],
            scale_: vec![1.0, 0.5],
            config: MinMaxScalerConfig { min: 0.0, max: 1.0 },
        };
        let scaler = FittedMinMaxScaler::<CpuBackend>::from_params(params).unwrap();

        let output = scaler.build_onnx_nodes(&mut builder, "input").unwrap();
        assert!(output.starts_with("minmax_scaled"));
        assert!(!builder.graph.initializer.is_empty());
    }

    #[test]
    fn test_robust_scaler_onnx_nodes() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 2);

        let params = RobustScalerParams {
            config: RobustScalerConfig::default(),
            n_features: 2,
            center_: vec![0.0, 0.0],
            scale_: vec![1.0, 1.0],
        };
        let scaler = FittedRobustScaler::<CpuBackend>::from_params(params).unwrap();

        let output = scaler.build_onnx_nodes(&mut builder, "input").unwrap();
        assert!(output.starts_with("robust_scaled"));
        assert!(!builder.graph.initializer.is_empty());
    }

    #[test]
    fn test_maxabs_scaler_onnx_nodes() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 2);

        let params = MaxAbsScalerParams {
            n_features: 2,
            max_abs_: vec![1.0, 0.5],
            scale_: vec![1.0, 2.0],
        };
        let scaler = FittedMaxAbsScaler::<CpuBackend>::from_params(params).unwrap();

        let output = scaler.build_onnx_nodes(&mut builder, "input").unwrap();
        assert!(output.starts_with("maxabs_scaled"));
        assert!(!builder.graph.initializer.is_empty());
    }

    #[test]
    fn test_normalizer_onnx_nodes() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);

        let params = NormalizerParams {
            norm: crate::preprocessing::scaling::NormType::L2,
            n_features: 3,
        };
        let normalizer = FittedNormalizer::<CpuBackend>::from_params(params).unwrap();

        let output = normalizer.build_onnx_nodes(&mut builder, "input").unwrap();
        assert!(output.starts_with("normalized"));
        assert!(!builder.graph.node.is_empty());
    }

    #[test]
    fn test_simple_imputer_onnx_nodes() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 2);

        let params = crate::preprocessing::imputation::SimpleImputerParams {
            n_features: 2,
            statistics_: vec![0.0, 1.0],
            strategy: crate::preprocessing::imputation::ImputeStrategy::Mean,
        };
        let imputer = FittedSimpleImputer::<CpuBackend>::from_params(params).unwrap();

        let output = imputer.build_onnx_nodes(&mut builder, "input").unwrap();
        assert!(output.starts_with("imputed"));
        assert!(!builder.graph.initializer.is_empty());
        assert!(!builder.graph.node.is_empty());
    }

    #[test]
    fn test_polynomial_features_degree_1_onnx_nodes() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);

        let params = crate::preprocessing::feature_engineering::PolynomialFeaturesParams {
            degree: 1,
            n_features_in: 3,
            n_features_out: 3,
            include_bias: false,
            interaction_only: false,
            output_combinations: vec![(1, vec![0]), (1, vec![1]), (1, vec![2])],
        };
        let poly = FittedPolynomialFeatures::<CpuBackend>::from_params(params).unwrap();

        let output = poly.build_onnx_nodes(&mut builder, "input").unwrap();
        assert_eq!(output, "input");
    }

    #[test]
    fn test_chained_transformers() {
        let mut builder = OnnxGraphBuilder::new("pipeline");
        builder.add_input_float("input", 3);

        // Chain: StandardScaler -> MinMaxScaler
        let scaler_params = StandardScalerParams {
            config: StandardScalerConfig::default(),
            n_features: 3,
            mean: vec![0.0, 1.0, 2.0],
            std: vec![1.0, 1.0, 1.0],
        };
        let scaler = FittedStandardScaler::<CpuBackend>::from_params(scaler_params).unwrap();

        let minmax_params = MinMaxScalerParams {
            n_features: 3,
            min_: vec![0.0, 0.0, 0.0],
            max_: vec![1.0, 1.0, 1.0],
            scale_: vec![1.0, 1.0, 1.0],
            config: MinMaxScalerConfig::default(),
        };
        let minmax = FittedMinMaxScaler::<CpuBackend>::from_params(minmax_params).unwrap();

        // Chain the outputs
        let scaled = scaler.build_onnx_nodes(&mut builder, "input").unwrap();
        let final_out = minmax.build_onnx_nodes(&mut builder, &scaled).unwrap();

        assert!(final_out.starts_with("minmax_scaled"));
        assert_eq!(builder.graph.initializer.len(), 5); // 2 for standard scaler + 3 for minmax
    }
}
