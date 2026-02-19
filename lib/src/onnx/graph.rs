//! ONNX graph builder utilities.
//!
//! Provides a builder pattern for constructing ONNX model graphs.

use super::error::OnnxError;
use super::proto::{
    AttributeProto, GraphProto, ModelProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TensorShapeProtoDimension, TypeProto, TypeProtoTensor, ValueInfoProto,
};
use crate::onnx::{DEFAULT_OPSET_VERSION, ML_OPSET_VERSION};
use prost::Message;
use std::collections::HashMap;

/// Builder for creating ONNX models.
pub struct OnnxGraphBuilder {
    /// Model being built.
    pub model: ModelProto,
    /// Graph being built.
    pub graph: GraphProto,
    /// Counter for generating unique names.
    name_counter: HashMap<String, usize>,
    /// Input names (in order).
    input_names: Vec<String>,
    /// Output names (in order).
    output_names: Vec<String>,
}

impl OnnxGraphBuilder {
    /// Create a new graph builder.
    pub fn new(model_name: &str) -> Self {
        let mut model = ModelProto::default();
        let mut graph = GraphProto::default();
        graph.name = model_name.to_string();

        // Add default opset
        model.opset_import.push(OperatorSetIdProto {
            domain: String::new(),
            version: DEFAULT_OPSET_VERSION,
        });

        Self {
            model,
            graph,
            name_counter: HashMap::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
        }
    }

    /// Add the ONNX ML operator set (for traditional ML operators).
    pub fn with_ml_opset(mut self) -> Self {
        self.model.opset_import.push(OperatorSetIdProto {
            domain: "ai.onnx.ml".to_string(),
            version: ML_OPSET_VERSION,
        });
        self
    }

    /// Generate a unique name for a node or tensor.
    pub fn unique_name(&mut self, base: &str) -> String {
        let count = self.name_counter.entry(base.to_string()).or_insert(0);
        *count += 1;
        format!("{}_{}", base, count)
    }

    /// Add an input tensor to the graph.
    pub fn add_input(&mut self, name: &str, elem_type: i32, shape: &[Option<i64>]) -> &mut Self {
        let dims: Vec<TensorShapeProtoDimension> = shape
            .iter()
            .map(|&d| match d {
                Some(v) => TensorShapeProtoDimension {
                    dim_value: v,
                    dim_param: String::new(),
                },
                None => TensorShapeProtoDimension {
                    dim_value: 0,
                    dim_param: "batch".to_string(),
                },
            })
            .collect();

        let input = ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TypeProtoTensor {
                    elem_type,
                    shape: Some(TensorShapeProto { dim: dims }),
                }),
            }),
        };

        self.graph.input.push(input);
        self.input_names.push(name.to_string());
        self
    }

    /// Add an input tensor with dynamic batch dimension.
    pub fn add_input_float(&mut self, name: &str, num_features: usize) -> &mut Self {
        // Shape: [batch_size, num_features] where batch_size is dynamic
        self.add_input(name, 1, &[None, Some(num_features as i64)])
    }

    /// Add an output tensor to the graph.
    pub fn add_output(&mut self, name: &str, elem_type: i32, shape: &[Option<i64>]) -> &mut Self {
        let dims: Vec<TensorShapeProtoDimension> = shape
            .iter()
            .map(|&d| match d {
                Some(v) => TensorShapeProtoDimension {
                    dim_value: v,
                    dim_param: String::new(),
                },
                None => TensorShapeProtoDimension {
                    dim_value: 0,
                    dim_param: "batch".to_string(),
                },
            })
            .collect();

        let output = ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TypeProtoTensor {
                    elem_type,
                    shape: Some(TensorShapeProto { dim: dims }),
                }),
            }),
        };

        self.graph.output.push(output);
        self.output_names.push(name.to_string());
        self
    }

    /// Add an output tensor with dynamic batch dimension.
    pub fn add_output_float(&mut self, name: &str, num_features: usize) -> &mut Self {
        // Shape: [batch_size, num_features] where batch_size is dynamic
        self.add_output(name, 1, &[None, Some(num_features as i64)])
    }

    /// Add a node to the graph.
    pub fn add_node(
        &mut self,
        op_type: &str,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: Vec<AttributeProto>,
    ) -> &mut Self {
        let name = self.unique_name(op_type);
        self.add_named_node(&name, op_type, inputs, outputs, attributes)
    }

    /// Add a named node to the graph.
    pub fn add_named_node(
        &mut self,
        name: &str,
        op_type: &str,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: Vec<AttributeProto>,
    ) -> &mut Self {
        use super::proto::NodeProto;
        let node = NodeProto {
            name: name.to_string(),
            op_type: op_type.to_string(),
            input: inputs,
            output: outputs,
            attribute: attributes,
            domain: String::new(),
        };
        self.graph.node.push(node);
        self
    }

    /// Add an initializer (constant tensor) to the graph.
    pub fn add_initializer(&mut self, tensor: TensorProto) -> &mut Self {
        self.graph.initializer.push(tensor);
        self
    }

    /// Add a float32 initializer tensor.
    pub fn add_float_initializer(&mut self, name: &str, dims: &[i64], data: &[f32]) -> &mut Self {
        let tensor = TensorProto::new_float(name, dims, data);
        self.add_initializer(tensor)
    }

    /// Add a float64 (double) initializer tensor.
    pub fn add_double_initializer(&mut self, name: &str, dims: &[i64], data: &[f64]) -> &mut Self {
        let tensor = TensorProto::new_double(name, dims, data);
        self.add_initializer(tensor)
    }

    /// Add an int64 initializer tensor.
    pub fn add_int64_initializer(&mut self, name: &str, dims: &[i64], data: &[i64]) -> &mut Self {
        let tensor = TensorProto::new_int64(name, dims, data);
        self.add_initializer(tensor)
    }

    /// Get the input names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get the output names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Build the model and serialize to bytes.
    pub fn build(mut self) -> Result<Vec<u8>, OnnxError> {
        self.model.graph = Some(self.graph);
        Ok(self.model.encode_to_vec())
    }

    /// Build and save to file.
    pub fn build_to_file(self, path: impl AsRef<std::path::Path>) -> Result<(), OnnxError> {
        let bytes = self.build()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

impl Default for OnnxGraphBuilder {
    fn default() -> Self {
        Self::new("model")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_builder_basic() {
        let mut builder = OnnxGraphBuilder::new("test_model");
        builder.add_input_float("input", 10);
        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_graph_builder_with_initializer() {
        let mut builder = OnnxGraphBuilder::new("test_model");
        builder.add_input_float("input", 2);
        builder.add_float_initializer("weights", &[2], &[1.0, 2.0]);
        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_unique_name() {
        let mut builder = OnnxGraphBuilder::new("test");
        let name1 = builder.unique_name("node");
        let name2 = builder.unique_name("node");
        let name3 = builder.unique_name("node");

        assert_eq!(name1, "node_1");
        assert_eq!(name2, "node_2");
        assert_eq!(name3, "node_3");
    }

    #[test]
    fn test_with_ml_opset() {
        let builder = OnnxGraphBuilder::new("test").with_ml_opset();
        assert_eq!(builder.model.opset_import.len(), 2);
        assert_eq!(builder.model.opset_import[1].domain, "ai.onnx.ml");
    }

    #[test]
    fn test_add_double_initializer() {
        let mut builder = OnnxGraphBuilder::new("test_model");
        builder.add_input_float("input", 2);
        builder.add_double_initializer("weights", &[2], &[1.0, 2.0]);
        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_add_int64_initializer() {
        let mut builder = OnnxGraphBuilder::new("test_model");
        builder.add_input_float("input", 2);
        builder.add_int64_initializer("indices", &[2], &[0, 1]);
        builder.add_output_float("output", 1);

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_input_output_names() {
        let mut builder = OnnxGraphBuilder::new("test_model");
        builder.add_input_float("input_a", 10);
        builder.add_input_float("input_b", 5);
        builder.add_output_float("output_a", 1);
        builder.add_output_float("output_b", 2);

        assert_eq!(builder.input_names(), &["input_a", "input_b"]);
        assert_eq!(builder.output_names(), &["output_a", "output_b"]);
    }

    #[test]
    fn test_build_to_file() {
        let mut builder = OnnxGraphBuilder::new("test_model");
        builder.add_input_float("input", 2);
        builder.add_float_initializer("weights", &[2], &[1.0, 2.0]);
        builder.add_output_float("output", 1);

        let temp_file = std::env::temp_dir().join("test_graph_build.onnx");
        builder.build_to_file(&temp_file).unwrap();

        let bytes = std::fs::read(&temp_file).unwrap();
        assert!(!bytes.is_empty());

        std::fs::remove_file(temp_file).ok();
    }
}
