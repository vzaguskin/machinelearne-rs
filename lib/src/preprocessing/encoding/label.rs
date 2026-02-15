//! Label encoding for 1D target labels.
//!
//! Maps target labels to integer indices (0, 1, 2, ...).

use crate::backend::{Backend, Tensor1D};
use crate::preprocessing::error::PreprocessingError;
use crate::serialization::SerializableParams;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Label encoder for 1D target labels.
///
/// Converts target labels to integer indices, suitable for classification
/// targets. Unlike OrdinalEncoder which works on 2D data, LabelEncoder
/// works on 1D arrays (a single target column).
///
/// # Example
/// ```ignore
/// use machinelearne_rs::preprocessing::LabelEncoder;
/// use machinelearne_rs::backend::CpuBackend;
///
/// // Target labels: [2, 0, 1, 0]
/// let labels = Tensor1D::new(vec![2.0, 0.0, 1.0, 0.0]);
///
/// let encoder = LabelEncoder::<CpuBackend>::new();
/// let fitted = encoder.fit(&labels)?;
///
/// // Encoded: [2, 0, 1, 0] (ordinal mapping)
/// let encoded = fitted.transform(&labels)?;
/// ```
#[derive(Clone, Debug)]
pub struct LabelEncoder<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> LabelEncoder<B> {
    /// Create a new LabelEncoder.
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    /// Fit the encoder to the labels and return the fitted encoder.
    pub fn fit(&self, labels: &Tensor1D<B>) -> Result<FittedLabelEncoder<B>, PreprocessingError> {
        let n = labels.len();

        if n == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit LabelEncoder on empty data".to_string(),
            ));
        }

        let label_vec = labels.to_vec();

        // Find unique classes
        let mut classes_set: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
        for &val in &label_vec {
            if !val.is_finite() {
                return Err(PreprocessingError::InvalidParameter(format!(
                    "LabelEncoder expects finite values, got {}",
                    val
                )));
            }
            classes_set.insert(val.round() as i32);
        }

        let classes_: Vec<f32> = classes_set.into_iter().map(|x| x as f32).collect();
        let n_classes = classes_.len();

        // Create mapping from class to index
        let mut class_to_idx: HashMap<i32, usize> = HashMap::new();
        for (idx, &class) in classes_.iter().enumerate() {
            class_to_idx.insert(class as i32, idx);
        }

        Ok(FittedLabelEncoder {
            classes_,
            class_to_idx,
            n_classes,
            _backend: PhantomData,
        })
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&self, labels: &Tensor1D<B>) -> Result<Tensor1D<B>, PreprocessingError> {
        let fitted = self.fit(labels)?;
        fitted.transform(labels)
    }
}

impl<B: Backend> Default for LabelEncoder<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable parameters for a fitted LabelEncoder.
#[derive(Clone, Serialize, Deserialize)]
pub struct LabelEncoderParams {
    /// Unique classes in sorted order.
    pub classes_: Vec<f32>,
    /// Number of unique classes.
    pub n_classes: usize,
}

// Note: We rely on the blanket impl of SerializableParams for Serialize + Deserialize

/// Fitted LabelEncoder ready for inference.
#[derive(Clone)]
pub struct FittedLabelEncoder<B: Backend> {
    /// Unique classes in sorted order.
    classes_: Vec<f32>,
    /// Mapping from class value to index.
    class_to_idx: HashMap<i32, usize>,
    /// Number of classes.
    n_classes: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedLabelEncoder<B> {
    /// Get the unique classes.
    pub fn classes(&self) -> &[f32] {
        &self.classes_
    }

    /// Get the number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Transform labels to encoded indices.
    pub fn transform(&self, labels: &Tensor1D<B>) -> Result<Tensor1D<B>, PreprocessingError> {
        let n = labels.len();
        if n == 0 {
            return Ok(Tensor1D::zeros(0));
        }

        let label_vec = labels.to_vec();
        let mut result = Vec::with_capacity(n);

        for &val in &label_vec {
            let key = val.round() as i32;
            match self.class_to_idx.get(&key) {
                Some(&idx) => result.push(idx as f64),
                None => {
                    return Err(PreprocessingError::InvalidParameter(format!(
                        "Unknown label value: {}",
                        val
                    )));
                }
            }
        }

        Ok(Tensor1D::new(result.iter().map(|&x| x as f32).collect()))
    }

    /// Inverse transform encoded indices back to original labels.
    pub fn inverse_transform(
        &self,
        indices: &Tensor1D<B>,
    ) -> Result<Tensor1D<B>, PreprocessingError> {
        let n = indices.len();
        if n == 0 {
            return Ok(Tensor1D::zeros(0));
        }

        let idx_vec = indices.to_vec();
        let mut result = Vec::with_capacity(n);

        for &idx in &idx_vec {
            let idx_usize = idx.round() as usize;
            if idx_usize >= self.n_classes {
                return Err(PreprocessingError::InvalidParameter(format!(
                    "Index {} out of bounds (max {})",
                    idx_usize,
                    self.n_classes - 1
                )));
            }
            result.push(self.classes_[idx_usize] as f64);
        }

        Ok(Tensor1D::new(result.iter().map(|&x| x as f32).collect()))
    }

    /// Extract parameters for serialization.
    pub fn extract_params(&self) -> LabelEncoderParams {
        LabelEncoderParams {
            classes_: self.classes_.clone(),
            n_classes: self.n_classes,
        }
    }

    /// Reconstruct from parameters.
    pub fn from_params(params: LabelEncoderParams) -> Result<Self, PreprocessingError> {
        let mut class_to_idx = HashMap::new();
        for (idx, &class) in params.classes_.iter().enumerate() {
            class_to_idx.insert(class as i32, idx);
        }

        Ok(FittedLabelEncoder {
            classes_: params.classes_,
            class_to_idx,
            n_classes: params.n_classes,
            _backend: PhantomData,
        })
    }

    /// Save to file.
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let params = self.extract_params();
        let bytes = params.to_bytes().map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Load from file.
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, PreprocessingError> {
        let bytes = std::fs::read(path)?;
        let params = LabelEncoderParams::from_bytes(&bytes)
            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
        Self::from_params(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_label_encoder_basic() {
        let labels = Tensor1D::<CpuBackend>::new(vec![2.0f32, 0.0, 1.0, 0.0]);

        let encoder = LabelEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&labels).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0.0f32, 1.0, 2.0]);

        let encoded = fitted.transform(&labels).unwrap();
        let vals = encoded.to_vec();

        // Classes: [0, 1, 2] -> indices: 0->0, 1->1, 2->2
        assert!((vals[0] - 2.0).abs() < 1e-6); // 2 -> idx 2
        assert!((vals[1] - 0.0).abs() < 1e-6); // 0 -> idx 0
        assert!((vals[2] - 1.0).abs() < 1e-6); // 1 -> idx 1
        assert!((vals[3] - 0.0).abs() < 1e-6); // 0 -> idx 0
    }

    #[test]
    fn test_label_encoder_non_contiguous() {
        let labels = Tensor1D::<CpuBackend>::new(vec![10.0f32, 5.0, 0.0]);

        let encoder = LabelEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&labels).unwrap();

        assert_eq!(fitted.classes(), &[0.0f32, 5.0, 10.0]);

        let encoded = fitted.transform(&labels).unwrap();
        let vals = encoded.to_vec();

        assert!((vals[0] - 2.0).abs() < 1e-6); // 10 -> idx 2
        assert!((vals[1] - 1.0).abs() < 1e-6); // 5 -> idx 1
        assert!((vals[2] - 0.0).abs() < 1e-6); // 0 -> idx 0
    }

    #[test]
    fn test_label_encoder_inverse() {
        let labels = Tensor1D::<CpuBackend>::new(vec![2.0f32, 0.0, 1.0, 0.0]);

        let encoder = LabelEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&labels).unwrap();

        let encoded = fitted.transform(&labels).unwrap();
        let recovered = fitted.inverse_transform(&encoded).unwrap();

        let orig = labels.to_vec();
        let rec = recovered.to_vec();

        for (o, r) in orig.iter().zip(rec.iter()) {
            assert!((o - r).abs() < 1e-6);
        }
    }

    #[test]
    fn test_label_encoder_unknown_error() {
        let train = Tensor1D::<CpuBackend>::new(vec![0.0f32, 1.0]);
        let test = Tensor1D::<CpuBackend>::new(vec![2.0f32]);

        let encoder = LabelEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&train).unwrap();

        let result = fitted.transform(&test);
        assert!(result.is_err());
    }

    #[test]
    fn test_label_encoder_serialization() {
        let labels = Tensor1D::<CpuBackend>::new(vec![2.0f32, 0.0, 1.0, 0.0]);

        let encoder = LabelEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&labels).unwrap();

        let temp_file = std::env::temp_dir().join("test_label.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedLabelEncoder::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_classes(), fitted.n_classes());
        assert_eq!(loaded.classes(), fitted.classes());

        let e1 = fitted.transform(&labels).unwrap();
        let e2 = loaded.transform(&labels).unwrap();

        let v1 = e1.to_vec();
        let v2 = e2.to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_label_encoder_empty_data() {
        let labels = Tensor1D::<CpuBackend>::new(vec![]);

        let encoder = LabelEncoder::<CpuBackend>::new();
        let result = encoder.fit(&labels);

        assert!(result.is_err());
    }

    #[test]
    fn test_label_encoder_len() {
        let labels = Tensor1D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0]);

        let encoder = LabelEncoder::<CpuBackend>::new();
        let fitted = encoder.fit(&labels).unwrap();

        assert_eq!(fitted.classes().len(), 3);
    }
}
