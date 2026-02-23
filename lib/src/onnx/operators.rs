//! ONNX operator tests.
//!
//! Tests for the ONNX operator methods on [`OnnxGraphBuilder`].

use super::graph::OnnxGraphBuilder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);
        builder.add_float_initializer("weights", &[1, 3], &[1.0, 2.0, 3.0]);
        builder.add_float_initializer("bias", &[1], &[0.5]);
        builder.gemm(
            "input",
            "weights",
            Some("bias"),
            false,
            true,
            1.0,
            1.0,
            "output",
        );
        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_basic_operators() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 2);
        builder.add_float_initializer("const", &[1, 2], &[1.0, 1.0]);

        builder.add("input", "const", "add_out");
        builder.sub("input", "const", "sub_out");
        builder.mul("input", "const", "mul_out");
        builder.div("input", "const", "div_out");

        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_reshape_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 6);
        builder.reshape("input", &[2, 3], "reshaped");
        builder.add_output_float("output", 3);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_reduce_operators() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 4);
        builder.reduce_mean("input", &[1], true, "mean_out");
        builder.reduce_sum("input", &[1], false, "sum_out");
        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_clip_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);
        builder.clip("input", Some(0.0), Some(1.0), "clipped");
        builder.add_output_float("output", 3);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);
        builder.unsqueeze("input", &[0], "expanded");
        builder.squeeze("expanded", &[0], "squeezed");
        builder.squeeze("input", &[], "empty_squeeze");
        builder.add_output_float("output", 3);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ml_operators() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);

        builder.scaler("input", &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], "scaled");
        builder.imputer("input", 0.0, &[1.0, 2.0, 3.0], "imputed");
        builder.normalizer("input", "L2", "normalized");
        builder.linear_regressor("input", &[1.0, 2.0, 3.0], &[0.5], "regressed");
        builder.one_hot_encoder("input", &[0, 1, 2], "encoded");

        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_matmul_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input_a", 3);
        builder.add_float_initializer("input_b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        builder.matmul("input_a", "input_b", "output");
        builder.add_output_float("output", 2);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_flatten_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 6);
        builder.flatten("input", 1, "flattened");
        builder.add_output_float("output", 6);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_concat_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input_a", 3);
        builder.add_input_float("input_b", 3);
        builder.concat(vec!["input_a", "input_b"], 1, "concatenated");
        builder.add_output_float("output", 6);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_pow_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);
        builder.add_float_initializer("exponent", &[1], &[2.0]);
        builder.pow("input", "exponent", "output");
        builder.add_output_float("output", 3);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_reciprocal_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);
        builder.reciprocal("input", "output");
        builder.add_output_float("output", 3);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_abs_operator() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);
        builder.abs("input", "output");
        builder.add_output_float("output", 3);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_chained_operators() {
        let mut builder = OnnxGraphBuilder::new("test");
        builder.add_input_float("input", 3);
        builder.add_float_initializer("offset", &[1, 3], &[1.0, 2.0, 3.0]);
        builder.add_float_initializer("scale", &[1, 3], &[0.5, 0.5, 0.5]);

        // Chain: (input - offset) * scale
        builder
            .sub("input", "offset", "subtracted")
            .mul("subtracted", "scale", "scaled");

        builder.add_output_float("output", 3);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }
}
