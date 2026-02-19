//! ONNX protobuf type definitions.
//!
//! This module contains the minimal ONNX protobuf types needed for model export.
//! Based on ONNX specification: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

use prost::Message;

/// ONNX ModelProto - the top-level model container.
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    /// ONNX version (required, must be 1).
    #[prost(int64, tag = "1")]
    pub ir_version: i64,

    /// Operator sets used by the model.
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,

    /// Model metadata.
    #[prost(string, tag = "4")]
    pub producer_name: String,

    #[prost(string, tag = "5")]
    pub producer_version: String,

    #[prost(string, tag = "7")]
    pub model_version: String,

    /// The main graph.
    #[prost(message, optional, tag = "100")]
    pub graph: Option<GraphProto>,
}

/// Operator set identifier.
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    /// Domain (empty string for default ONNX domain).
    #[prost(string, tag = "1")]
    pub domain: String,

    /// Version.
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// Graph definition.
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    /// Graph name.
    #[prost(string, tag = "1")]
    pub name: String,

    /// Input tensors.
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,

    /// Output tensors.
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,

    /// Computation nodes.
    #[prost(message, repeated, tag = "13")]
    pub node: Vec<NodeProto>,

    /// Initializers (constant tensors).
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
}

/// Node in the computation graph.
#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    /// Input names.
    #[prost(string, repeated, tag = "4")]
    pub input: Vec<String>,

    /// Output names.
    #[prost(string, repeated, tag = "5")]
    pub output: Vec<String>,

    /// Node name (optional).
    #[prost(string, tag = "3")]
    pub name: String,

    /// Operator type.
    #[prost(string, tag = "1")]
    pub op_type: String,

    /// Operator domain (empty for default).
    #[prost(string, tag = "7")]
    pub domain: String,

    /// Operator attributes.
    #[prost(message, repeated, tag = "6")]
    pub attribute: Vec<AttributeProto>,
}

/// Tensor value.
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    /// Name.
    #[prost(string, tag = "1")]
    pub name: String,

    /// Type.
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

/// Type specification.
#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    /// Tensor type.
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TypeProtoTensor>,
}

/// Tensor type specification.
#[derive(Clone, PartialEq, Message)]
pub struct TypeProtoTensor {
    /// Element type (1=float, 2=uint8, 3=int8, etc.).
    #[prost(int32, tag = "1")]
    pub elem_type: i32,

    /// Shape.
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

/// Tensor shape.
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    /// Dimensions.
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeProtoDimension>,
}

/// Dimension in a tensor shape.
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProtoDimension {
    /// Dimension value (for fixed dimensions).
    #[prost(int64, tag = "1")]
    pub dim_value: i64,

    /// Dimension name (for symbolic dimensions).
    #[prost(string, tag = "2")]
    pub dim_param: String,
}

/// Tensor data.
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    /// Data type (1=float32, 2=uint8, 3=int8, 6=int32, 7=int64, etc.).
    #[prost(int32, tag = "1")]
    pub data_type: i32,

    /// Shape dimensions.
    #[prost(int64, repeated, tag = "2")]
    pub dims: Vec<i64>,

    /// Tensor name.
    #[prost(string, tag = "3")]
    pub name: String,

    /// Float data (for float type).
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,

    /// Int64 data (for int64 type).
    #[prost(int64, repeated, tag = "6")]
    pub int64_data: Vec<i64>,

    /// Raw data bytes (for raw_data field).
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,

    /// Double data (for double type).
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
}

/// Attribute on a node.
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    /// Attribute name.
    #[prost(string, tag = "1")]
    pub name: String,

    /// Attribute value type.
    #[prost(enumeration = "AttributeType", tag = "20")]
    pub r#type: i32,

    /// Float value.
    #[prost(float, tag = "2")]
    pub f: f32,

    /// Int value.
    #[prost(int64, tag = "3")]
    pub i: i64,

    /// String value.
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,

    /// Tensor value.
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,

    /// Floats value.
    #[prost(float, repeated, tag = "6")]
    pub floats: Vec<f32>,

    /// Ints value.
    #[prost(int64, repeated, tag = "7")]
    pub ints: Vec<i64>,
}

/// Attribute value types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(i32)]
pub enum AttributeType {
    #[default]
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
}

impl From<AttributeType> for i32 {
    fn from(value: AttributeType) -> Self {
        value as i32
    }
}

impl TryFrom<i32> for AttributeType {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(AttributeType::Undefined),
            1 => Ok(AttributeType::Float),
            2 => Ok(AttributeType::Int),
            3 => Ok(AttributeType::String),
            4 => Ok(AttributeType::Tensor),
            5 => Ok(AttributeType::Graph),
            6 => Ok(AttributeType::Floats),
            7 => Ok(AttributeType::Ints),
            8 => Ok(AttributeType::Strings),
            9 => Ok(AttributeType::Tensors),
            10 => Ok(AttributeType::Graphs),
            _ => Err(()),
        }
    }
}

/// Tensor data types.
pub mod tensor_data_type {
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const UINT16: i32 = 4;
    pub const INT16: i32 = 5;
    pub const INT32: i32 = 6;
    pub const INT64: i32 = 7;
    pub const STRING: i32 = 8;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const DOUBLE: i32 = 11;
    pub const UINT32: i32 = 12;
    pub const UINT64: i32 = 13;
    pub const COMPLEX64: i32 = 14;
    pub const COMPLEX128: i32 = 15;
    pub const BFLOAT16: i32 = 16;
}

impl TensorProto {
    /// Create a new float tensor with the given name, shape, and data.
    pub fn new_float(name: impl Into<String>, dims: &[i64], data: &[f32]) -> Self {
        // Use raw_data for better compatibility
        let raw_data: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        Self {
            data_type: tensor_data_type::FLOAT,
            dims: dims.to_vec(),
            name: name.into(),
            raw_data,
            float_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
        }
    }

    /// Create a new int64 tensor with the given name, shape, and data.
    pub fn new_int64(name: impl Into<String>, dims: &[i64], data: &[i64]) -> Self {
        // Use raw_data for better compatibility
        let raw_data: Vec<u8> = data.iter().flat_map(|&i| i.to_le_bytes()).collect();

        Self {
            data_type: tensor_data_type::INT64,
            dims: dims.to_vec(),
            name: name.into(),
            raw_data,
            float_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
        }
    }

    /// Create a new double (f64) tensor with the given name, shape, and data.
    pub fn new_double(name: impl Into<String>, dims: &[i64], data: &[f64]) -> Self {
        let raw_data: Vec<u8> = data.iter().flat_map(|&d| d.to_le_bytes()).collect();

        Self {
            data_type: tensor_data_type::DOUBLE,
            dims: dims.to_vec(),
            name: name.into(),
            raw_data,
            float_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
        }
    }
}

impl AttributeProto {
    /// Create a float attribute.
    pub fn float(name: impl Into<String>, value: f32) -> Self {
        Self {
            name: name.into(),
            r#type: AttributeType::Float as i32,
            f: value,
            i: 0,
            s: Vec::new(),
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
        }
    }

    /// Create an int attribute.
    pub fn int(name: impl Into<String>, value: i64) -> Self {
        Self {
            name: name.into(),
            r#type: AttributeType::Int as i32,
            f: 0.0,
            i: value,
            s: Vec::new(),
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
        }
    }

    /// Create a string attribute.
    pub fn string(name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        Self {
            name: name.into(),
            r#type: AttributeType::String as i32,
            f: 0.0,
            i: 0,
            s: value.into(),
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
        }
    }

    /// Create an ints attribute.
    pub fn ints(name: impl Into<String>, values: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            r#type: AttributeType::Ints as i32,
            f: 0.0,
            i: 0,
            s: Vec::new(),
            t: None,
            floats: Vec::new(),
            ints: values,
        }
    }

    /// Create a floats attribute.
    pub fn floats(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            r#type: AttributeType::Floats as i32,
            f: 0.0,
            i: 0,
            s: Vec::new(),
            t: None,
            floats: values,
            ints: Vec::new(),
        }
    }
}
