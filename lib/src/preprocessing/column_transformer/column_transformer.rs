//! ColumnTransformer implementation.
//!
//! Applies different transformers to different column subsets and concatenates results.

use crate::backend::{Backend, Tensor2D};
use crate::preprocessing::encoding::{
    FittedOneHotEncoder, FittedOrdinalEncoder, OneHotEncoder, OneHotEncoderParams, OrdinalEncoder,
    OrdinalEncoderParams,
};
use crate::preprocessing::error::PreprocessingError;
use crate::preprocessing::imputation::{FittedSimpleImputer, SimpleImputer, SimpleImputerParams};
use crate::preprocessing::pipeline::pipeline::{FittedPipeline, Pipeline, PipelineStepEnum};
use crate::preprocessing::scaling::{
    FittedMaxAbsScaler, FittedMinMaxScaler, FittedNormalizer, FittedRobustScaler,
    FittedStandardScaler, MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams,
    Normalizer, NormalizerParams, RobustScaler, RobustScalerParams, StandardScaler,
    StandardScalerParams,
};
use crate::preprocessing::traits::{FittedTransformer, Transformer};
use crate::serialization::SerializableParams;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Range;

/// Specifies which columns a transformer should be applied to.
#[derive(Clone, Debug)]
pub enum ColumnSpec {
    /// Apply to specific column indices.
    Indices(Vec<usize>),
    /// Apply to a range of columns.
    Range(Range<usize>),
    /// Apply to all columns.
    All,
}

impl ColumnSpec {
    /// Resolve the column spec to actual column indices.
    fn resolve(&self, n_features: usize) -> Vec<usize> {
        match self {
            ColumnSpec::Indices(indices) => indices.clone(),
            ColumnSpec::Range(range) => range.clone().collect(),
            ColumnSpec::All => (0..n_features).collect(),
        }
    }
}

/// Enum of unfitted transformers that can be used in a ColumnTransformer.
#[derive(Clone)]
pub enum ColumnTransformerStep<B: Backend> {
    StandardScaler(StandardScaler<B>),
    MinMaxScaler(MinMaxScaler<B>),
    RobustScaler(RobustScaler<B>),
    MaxAbsScaler(MaxAbsScaler<B>),
    Normalizer(Normalizer<B>),
    SimpleImputer(SimpleImputer<B>),
    OneHotEncoder(OneHotEncoder<B>),
    OrdinalEncoder(OrdinalEncoder<B>),
    Pipeline(Pipeline<B>),
}

/// Enum of fitted transformers for ColumnTransformer.
#[derive(Clone)]
pub enum FittedColumnTransformerStep<B: Backend> {
    StandardScaler(FittedStandardScaler<B>),
    MinMaxScaler(FittedMinMaxScaler<B>),
    RobustScaler(FittedRobustScaler<B>),
    MaxAbsScaler(FittedMaxAbsScaler<B>),
    Normalizer(FittedNormalizer<B>),
    SimpleImputer(FittedSimpleImputer<B>),
    OneHotEncoder(FittedOneHotEncoder<B>),
    OrdinalEncoder(FittedOrdinalEncoder<B>),
    Pipeline(FittedPipeline<B>),
}

impl<B: Backend> FittedColumnTransformerStep<B> {
    /// Transform the data.
    fn transform(&self, data: &Tensor2D<B>) -> Result<Tensor2D<B>, PreprocessingError> {
        match self {
            FittedColumnTransformerStep::StandardScaler(t) => t.transform(data),
            FittedColumnTransformerStep::MinMaxScaler(t) => t.transform(data),
            FittedColumnTransformerStep::RobustScaler(t) => t.transform(data),
            FittedColumnTransformerStep::MaxAbsScaler(t) => t.transform(data),
            FittedColumnTransformerStep::Normalizer(t) => t.transform(data),
            FittedColumnTransformerStep::SimpleImputer(t) => t.transform(data),
            FittedColumnTransformerStep::OneHotEncoder(t) => t.transform(data),
            FittedColumnTransformerStep::OrdinalEncoder(t) => t.transform(data),
            FittedColumnTransformerStep::Pipeline(t) => t.transform(data),
        }
    }

    /// Get the step name.
    fn step_name(&self) -> &'static str {
        match self {
            FittedColumnTransformerStep::StandardScaler(_) => "StandardScaler",
            FittedColumnTransformerStep::MinMaxScaler(_) => "MinMaxScaler",
            FittedColumnTransformerStep::RobustScaler(_) => "RobustScaler",
            FittedColumnTransformerStep::MaxAbsScaler(_) => "MaxAbsScaler",
            FittedColumnTransformerStep::Normalizer(_) => "Normalizer",
            FittedColumnTransformerStep::SimpleImputer(_) => "SimpleImputer",
            FittedColumnTransformerStep::OneHotEncoder(_) => "OneHotEncoder",
            FittedColumnTransformerStep::OrdinalEncoder(_) => "OrdinalEncoder",
            FittedColumnTransformerStep::Pipeline(_) => "Pipeline",
        }
    }

    /// Get the number of output features.
    fn n_features_out(&self) -> usize {
        match self {
            FittedColumnTransformerStep::StandardScaler(t) => t.n_features_in(),
            FittedColumnTransformerStep::MinMaxScaler(t) => t.n_features_in(),
            FittedColumnTransformerStep::RobustScaler(t) => t.n_features_in(),
            FittedColumnTransformerStep::MaxAbsScaler(t) => t.n_features_in(),
            FittedColumnTransformerStep::Normalizer(t) => t.n_features_in(),
            FittedColumnTransformerStep::SimpleImputer(t) => t.n_features_in(),
            FittedColumnTransformerStep::OneHotEncoder(t) => t.n_features_out(),
            FittedColumnTransformerStep::OrdinalEncoder(t) => t.n_features_in(),
            FittedColumnTransformerStep::Pipeline(t) => t.n_features_in(),
        }
    }
}

/// Fit a column transformer step from an unfitted step.
fn fit_step<B: Backend>(
    step: &ColumnTransformerStep<B>,
    data: &Tensor2D<B>,
) -> Result<FittedColumnTransformerStep<B>, PreprocessingError> {
    match step {
        ColumnTransformerStep::StandardScaler(t) => {
            t.fit(data).map(FittedColumnTransformerStep::StandardScaler)
        }
        ColumnTransformerStep::MinMaxScaler(t) => {
            t.fit(data).map(FittedColumnTransformerStep::MinMaxScaler)
        }
        ColumnTransformerStep::RobustScaler(t) => {
            t.fit(data).map(FittedColumnTransformerStep::RobustScaler)
        }
        ColumnTransformerStep::MaxAbsScaler(t) => {
            t.fit(data).map(FittedColumnTransformerStep::MaxAbsScaler)
        }
        ColumnTransformerStep::Normalizer(t) => {
            t.fit(data).map(FittedColumnTransformerStep::Normalizer)
        }
        ColumnTransformerStep::SimpleImputer(t) => {
            t.fit(data).map(FittedColumnTransformerStep::SimpleImputer)
        }
        ColumnTransformerStep::OneHotEncoder(t) => {
            t.fit(data).map(FittedColumnTransformerStep::OneHotEncoder)
        }
        ColumnTransformerStep::OrdinalEncoder(t) => {
            t.fit(data).map(FittedColumnTransformerStep::OrdinalEncoder)
        }
        ColumnTransformerStep::Pipeline(t) => {
            t.fit(data).map(FittedColumnTransformerStep::Pipeline)
        }
    }
}

/// ColumnTransformer applies different transformers to different columns.
///
/// This is useful when you have heterogeneous data and want to apply
/// different preprocessing to different feature subsets (e.g., scale
/// numerical features, one-hot encode categorical features).
///
/// # Example
/// ```ignore
/// use machinelearne_rs::preprocessing::{
///     ColumnTransformer, ColumnSpec, StandardScaler, OneHotEncoder, Transformer
/// };
/// use machinelearne_rs::backend::CpuBackend;
///
/// // Columns: [age, income, city_code]
/// // Scale numerical cols [0, 1], one-hot encode col [2]
/// let ct = ColumnTransformer::<CpuBackend>::new()
///     .add(StandardScaler::new(), ColumnSpec::Indices(vec![0, 1]))
///     .add(OneHotEncoder::new(), ColumnSpec::Indices(vec![2]));
///
/// let fitted = ct.fit(&data)?;
/// let transformed = fitted.transform(&data)?;
/// ```
#[derive(Clone)]
pub struct ColumnTransformer<B: Backend> {
    steps: Vec<(ColumnSpec, ColumnTransformerStep<B>)>,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for ColumnTransformer<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ColumnTransformer<B> {
    /// Create a new empty ColumnTransformer.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            _backend: PhantomData,
        }
    }

    /// Add a StandardScaler for specified columns.
    pub fn add_standard_scaler(mut self, scaler: StandardScaler<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::StandardScaler(scaler)));
        self
    }

    /// Add a MinMaxScaler for specified columns.
    pub fn add_minmax_scaler(mut self, scaler: MinMaxScaler<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::MinMaxScaler(scaler)));
        self
    }

    /// Add a RobustScaler for specified columns.
    pub fn add_robust_scaler(mut self, scaler: RobustScaler<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::RobustScaler(scaler)));
        self
    }

    /// Add a MaxAbsScaler for specified columns.
    pub fn add_maxabs_scaler(mut self, scaler: MaxAbsScaler<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::MaxAbsScaler(scaler)));
        self
    }

    /// Add a Normalizer for specified columns.
    pub fn add_normalizer(mut self, normalizer: Normalizer<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::Normalizer(normalizer)));
        self
    }

    /// Add a SimpleImputer for specified columns.
    pub fn add_simple_imputer(mut self, imputer: SimpleImputer<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::SimpleImputer(imputer)));
        self
    }

    /// Add a OneHotEncoder for specified columns.
    pub fn add_one_hot_encoder(mut self, encoder: OneHotEncoder<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::OneHotEncoder(encoder)));
        self
    }

    /// Add an OrdinalEncoder for specified columns.
    pub fn add_ordinal_encoder(mut self, encoder: OrdinalEncoder<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::OrdinalEncoder(encoder)));
        self
    }

    /// Add a Pipeline for specified columns.
    pub fn add_pipeline(mut self, pipeline: Pipeline<B>, spec: ColumnSpec) -> Self {
        self.steps
            .push((spec, ColumnTransformerStep::Pipeline(pipeline)));
        self
    }

    /// Add a generic step.
    pub fn add(mut self, step: ColumnTransformerStep<B>, spec: ColumnSpec) -> Self {
        self.steps.push((spec, step));
        self
    }

    /// Get the number of transformer steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Serializable parameters for fitted column transformer step.
#[derive(Clone, Serialize, Deserialize)]
pub struct StepParams {
    /// Column indices this step was applied to.
    pub columns: Vec<usize>,
    /// Step type name.
    pub step_type: String,
    /// Serialized step parameters.
    pub params: Vec<u8>,
}

/// Serializable parameters for a fitted ColumnTransformer.
#[derive(Clone, Serialize, Deserialize)]
pub struct ColumnTransformerParams {
    /// Number of input features.
    pub n_features_in: usize,
    /// Number of output features.
    pub n_features_out: usize,
    /// Step parameters.
    pub steps: Vec<StepParams>,
}

/// Fitted ColumnTransformer ready for inference.
#[derive(Clone)]
pub struct FittedColumnTransformer<B: Backend> {
    /// Fitted steps with their column indices.
    fitted_steps: Vec<(Vec<usize>, FittedColumnTransformerStep<B>)>,
    /// Number of input features.
    n_features_in: usize,
    /// Number of output features.
    n_features_out: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> FittedColumnTransformer<B> {
    /// Get the number of input features.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Get the number of output features.
    pub fn n_features_out(&self) -> usize {
        self.n_features_out
    }

    /// Get step names.
    pub fn step_names(&self) -> Vec<(&'static str, &[usize])> {
        self.fitted_steps
            .iter()
            .map(|(cols, step)| (step.step_name(), cols.as_slice()))
            .collect()
    }
}

impl<B: Backend> Transformer<B> for ColumnTransformer<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = ColumnTransformerParams;
    type Fitted = FittedColumnTransformer<B>;

    fn fit(&self, data: &Self::Input) -> Result<Self::Fitted, PreprocessingError> {
        let (rows, cols) = data.shape();

        if rows == 0 {
            return Err(PreprocessingError::EmptyData(
                "Cannot fit ColumnTransformer on empty data".to_string(),
            ));
        }

        if self.steps.is_empty() {
            return Err(PreprocessingError::InvalidParameter(
                "Cannot fit empty ColumnTransformer".to_string(),
            ));
        }

        let mut fitted_steps = Vec::with_capacity(self.steps.len());
        let mut n_features_out = 0;

        for (spec, step) in &self.steps {
            let columns = spec.resolve(cols);

            // Validate columns
            for &col in &columns {
                if col >= cols {
                    return Err(PreprocessingError::InvalidParameter(format!(
                        "Column index {} out of bounds (max {})",
                        col,
                        cols - 1
                    )));
                }
            }

            // Extract columns
            let col_data = extract_columns(data, &columns)?;

            // Fit transformer
            let fitted = fit_step(step, &col_data)?;

            n_features_out += fitted.n_features_out();
            fitted_steps.push((columns, fitted));
        }

        Ok(FittedColumnTransformer {
            fitted_steps,
            n_features_in: cols,
            n_features_out,
            _backend: PhantomData,
        })
    }

    fn fit_transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let fitted = self.fit(data)?;
        fitted.transform(data)
    }
}

impl<B: Backend> FittedTransformer<B> for FittedColumnTransformer<B> {
    type Input = Tensor2D<B>;
    type Output = Tensor2D<B>;
    type Params = ColumnTransformerParams;

    fn transform(&self, data: &Self::Input) -> Result<Self::Output, PreprocessingError> {
        let (rows, cols) = data.shape();

        if cols != self.n_features_in {
            return Err(PreprocessingError::FeatureMismatch {
                expected_features: self.n_features_in,
                got_features: cols,
            });
        }

        if rows == 0 {
            return Ok(Tensor2D::zeros(0, self.n_features_out));
        }

        // Transform each step and collect outputs
        let mut transformed_outputs = Vec::with_capacity(self.fitted_steps.len());

        for (columns, step) in &self.fitted_steps {
            let col_data = extract_columns(data, columns)?;
            let transformed = step.transform(&col_data)?;
            transformed_outputs.push(transformed);
        }

        // Concatenate all outputs horizontally
        hcat_tensors(&transformed_outputs)
    }

    fn inverse_transform(&self, _data: &Self::Output) -> Result<Self::Input, PreprocessingError> {
        Err(PreprocessingError::InvalidParameter(
            "ColumnTransformer does not support inverse_transform".to_string(),
        ))
    }

    fn extract_params(&self) -> Self::Params {
        let steps = self
            .fitted_steps
            .iter()
            .map(|(columns, step)| {
                let (step_type, params) = match step {
                    FittedColumnTransformerStep::StandardScaler(t) => (
                        "StandardScaler".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::MinMaxScaler(t) => (
                        "MinMaxScaler".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::RobustScaler(t) => (
                        "RobustScaler".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::MaxAbsScaler(t) => (
                        "MaxAbsScaler".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::Normalizer(t) => (
                        "Normalizer".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::SimpleImputer(t) => (
                        "SimpleImputer".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::OneHotEncoder(t) => (
                        "OneHotEncoder".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::OrdinalEncoder(t) => (
                        "OrdinalEncoder".to_string(),
                        t.extract_params().to_bytes().unwrap(),
                    ),
                    FittedColumnTransformerStep::Pipeline(t) => {
                        // Pipeline needs special handling
                        let mut step_params = Vec::new();
                        for s in t.steps() {
                            let (name, bytes) = match s {
                                PipelineStepEnum::StandardScaler(st) => {
                                    ("StandardScaler", st.extract_params().to_bytes().unwrap())
                                }
                                PipelineStepEnum::MinMaxScaler(st) => {
                                    ("MinMaxScaler", st.extract_params().to_bytes().unwrap())
                                }
                                PipelineStepEnum::RobustScaler(st) => {
                                    ("RobustScaler", st.extract_params().to_bytes().unwrap())
                                }
                                PipelineStepEnum::MaxAbsScaler(st) => {
                                    ("MaxAbsScaler", st.extract_params().to_bytes().unwrap())
                                }
                                PipelineStepEnum::Normalizer(st) => {
                                    ("Normalizer", st.extract_params().to_bytes().unwrap())
                                }
                                PipelineStepEnum::SimpleImputer(st) => {
                                    ("SimpleImputer", st.extract_params().to_bytes().unwrap())
                                }
                                PipelineStepEnum::OneHotEncoder(st) => {
                                    ("OneHotEncoder", st.extract_params().to_bytes().unwrap())
                                }
                                PipelineStepEnum::OrdinalEncoder(st) => {
                                    ("OrdinalEncoder", st.extract_params().to_bytes().unwrap())
                                }
                            };
                            step_params.push((name.to_string(), bytes));
                        }
                        (
                            "Pipeline".to_string(),
                            bincode::serialize(&step_params).unwrap(),
                        )
                    }
                };
                StepParams {
                    columns: columns.clone(),
                    step_type,
                    params,
                }
            })
            .collect();

        ColumnTransformerParams {
            n_features_in: self.n_features_in,
            n_features_out: self.n_features_out,
            steps,
        }
    }

    fn from_params(params: Self::Params) -> Result<Self, PreprocessingError> {
        let mut fitted_steps = Vec::with_capacity(params.steps.len());

        for step_params in params.steps {
            let step = match step_params.step_type.as_str() {
                "StandardScaler" => {
                    let p: StandardScalerParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::StandardScaler(FittedStandardScaler::from_params(
                        p,
                    )?)
                }
                "MinMaxScaler" => {
                    let p: MinMaxScalerParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::MinMaxScaler(FittedMinMaxScaler::from_params(p)?)
                }
                "RobustScaler" => {
                    let p: RobustScalerParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::RobustScaler(FittedRobustScaler::from_params(p)?)
                }
                "MaxAbsScaler" => {
                    let p: MaxAbsScalerParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::MaxAbsScaler(FittedMaxAbsScaler::from_params(p)?)
                }
                "Normalizer" => {
                    let p: NormalizerParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::Normalizer(FittedNormalizer::from_params(p)?)
                }
                "SimpleImputer" => {
                    let p: SimpleImputerParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::SimpleImputer(FittedSimpleImputer::from_params(p)?)
                }
                "OneHotEncoder" => {
                    let p: OneHotEncoderParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::OneHotEncoder(FittedOneHotEncoder::from_params(p)?)
                }
                "OrdinalEncoder" => {
                    let p: OrdinalEncoderParams = bincode::deserialize(&step_params.params)
                        .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    FittedColumnTransformerStep::OrdinalEncoder(FittedOrdinalEncoder::from_params(
                        p,
                    )?)
                }
                "Pipeline" => {
                    let inner_steps: Vec<(String, Vec<u8>)> =
                        bincode::deserialize(&step_params.params)
                            .map_err(|e| PreprocessingError::SerializationError(e.to_string()))?;
                    let mut steps = Vec::new();
                    for (name, bytes) in inner_steps {
                        let step = match name.as_str() {
                            "StandardScaler" => {
                                let p: StandardScalerParams = bincode::deserialize(&bytes)
                                    .map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::StandardScaler(FittedStandardScaler::from_params(
                                    p,
                                )?)
                            }
                            "MinMaxScaler" => {
                                let p: MinMaxScalerParams =
                                    bincode::deserialize(&bytes).map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::MinMaxScaler(FittedMinMaxScaler::from_params(p)?)
                            }
                            "RobustScaler" => {
                                let p: RobustScalerParams =
                                    bincode::deserialize(&bytes).map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::RobustScaler(FittedRobustScaler::from_params(p)?)
                            }
                            "MaxAbsScaler" => {
                                let p: MaxAbsScalerParams =
                                    bincode::deserialize(&bytes).map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::MaxAbsScaler(FittedMaxAbsScaler::from_params(p)?)
                            }
                            "Normalizer" => {
                                let p: NormalizerParams =
                                    bincode::deserialize(&bytes).map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::Normalizer(FittedNormalizer::from_params(p)?)
                            }
                            "SimpleImputer" => {
                                let p: SimpleImputerParams =
                                    bincode::deserialize(&bytes).map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::SimpleImputer(FittedSimpleImputer::from_params(
                                    p,
                                )?)
                            }
                            "OneHotEncoder" => {
                                let p: OneHotEncoderParams =
                                    bincode::deserialize(&bytes).map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::OneHotEncoder(FittedOneHotEncoder::from_params(
                                    p,
                                )?)
                            }
                            "OrdinalEncoder" => {
                                let p: OrdinalEncoderParams = bincode::deserialize(&bytes)
                                    .map_err(|e| {
                                        PreprocessingError::SerializationError(e.to_string())
                                    })?;
                                PipelineStepEnum::OrdinalEncoder(FittedOrdinalEncoder::from_params(
                                    p,
                                )?)
                            }
                            _ => {
                                return Err(PreprocessingError::SerializationError(format!(
                                    "Unknown step type: {}",
                                    name
                                )))
                            }
                        };
                        steps.push(step);
                    }
                    // Get n_features from the first step
                    let n_features = steps.first().map(|s| s.n_features_in()).unwrap_or(0);
                    FittedColumnTransformerStep::Pipeline(FittedPipeline::from_steps(
                        steps, n_features,
                    ))
                }
                _ => {
                    return Err(PreprocessingError::SerializationError(format!(
                        "Unknown step type: {}",
                        step_params.step_type
                    )))
                }
            };
            fitted_steps.push((step_params.columns, step));
        }

        Ok(FittedColumnTransformer {
            fitted_steps,
            n_features_in: params.n_features_in,
            n_features_out: params.n_features_out,
            _backend: PhantomData,
        })
    }

    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

/// Extract specified columns from a 2D tensor.
fn extract_columns<B: Backend>(
    data: &Tensor2D<B>,
    columns: &[usize],
) -> Result<Tensor2D<B>, PreprocessingError> {
    if columns.is_empty() {
        let (rows, _) = data.shape();
        return Ok(Tensor2D::zeros(rows, 0));
    }

    // Use the backend method to select columns
    let inner = B::select_columns_2d(&data.data, columns);
    Ok(Tensor2D {
        data: inner,
        backend: PhantomData,
    })
}

/// Horizontally concatenate tensors.
fn hcat_tensors<B: Backend>(tensors: &[Tensor2D<B>]) -> Result<Tensor2D<B>, PreprocessingError> {
    if tensors.is_empty() {
        return Err(PreprocessingError::InvalidParameter(
            "Cannot concatenate empty slice of tensors".to_string(),
        ));
    }

    let inner_tensors: Vec<_> = tensors.iter().map(|t| t.data.clone()).collect();
    let result = B::hcat_2d(&inner_tensors)?;
    Ok(Tensor2D {
        data: result,
        backend: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::preprocessing::imputation::ImputeStrategy;
    use crate::preprocessing::scaling::NormType;

    #[test]
    fn test_column_transformer_basic() {
        // [[1, 10], [2, 20], [3, 30]]
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::Indices(vec![0]))
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::Indices(vec![1]));

        let fitted = ct.fit(&data).unwrap();
        assert_eq!(fitted.n_features_in(), 2);
        assert_eq!(fitted.n_features_out(), 2);

        let transformed = fitted.transform(&data).unwrap();
        assert_eq!(transformed.shape(), (3, 2));
    }

    #[test]
    fn test_column_transformer_one_hot() {
        // [[0], [1], [0]]
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 0.0], 3, 1);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_one_hot_encoder(OneHotEncoder::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        assert_eq!(fitted.n_features_out(), 2); // 2 categories

        let transformed = fitted.transform(&data).unwrap();
        assert_eq!(transformed.shape(), (3, 2));
    }

    #[test]
    fn test_column_transformer_mixed() {
        // [[1, 10, 0], [2, 20, 1], [3, 30, 0]] - cols 0,1 numeric, col 2 categorical
        let data = Tensor2D::<CpuBackend>::new(
            vec![1.0f32, 10.0, 0.0, 2.0, 20.0, 1.0, 3.0, 30.0, 0.0],
            3,
            3,
        );

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::Range(0..2))
            .add_one_hot_encoder(OneHotEncoder::new(), ColumnSpec::Indices(vec![2]));

        let fitted = ct.fit(&data).unwrap();
        // StandardScaler outputs 2 cols, OneHotEncoder outputs 2 cols = 4 total
        assert_eq!(fitted.n_features_out(), 4);

        let transformed = fitted.transform(&data).unwrap();
        assert_eq!(transformed.shape(), (3, 4));
    }

    #[test]
    fn test_column_transformer_with_pipeline() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2);

        let pipeline = Pipeline::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new())
            .add_minmax_scaler(MinMaxScaler::new().with_range(0.0, 1.0));

        let ct = ColumnTransformer::<CpuBackend>::new().add_pipeline(pipeline, ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Check values are in [0, 1] range after MinMaxScaler
        let vals = transformed.ravel().to_vec();
        for &v in &vals {
            assert!(v >= -1e-6 && v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_column_transformer_serialization() {
        let data = Tensor2D::<CpuBackend>::new(
            vec![1.0f32, 10.0, 0.0, 2.0, 20.0, 1.0, 3.0, 30.0, 0.0],
            3,
            3,
        );

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::Indices(vec![0, 1]))
            .add_one_hot_encoder(OneHotEncoder::new(), ColumnSpec::Indices(vec![2]));

        let fitted = ct.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_column_transformer.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedColumnTransformer::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());
        assert_eq!(loaded.n_features_out(), fitted.n_features_out());

        // Compare transform results
        let t1 = fitted.transform(&data).unwrap();
        let t2 = loaded.transform(&data).unwrap();

        let v1 = t1.ravel().to_vec();
        let v2 = t2.ravel().to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_column_transformer_feature_mismatch() {
        let train = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0], 2, 2);
        let test = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0], 1, 3);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);

        let fitted = ct.fit(&train).unwrap();

        let result = fitted.transform(&test);
        assert!(matches!(
            result,
            Err(PreprocessingError::FeatureMismatch { .. })
        ));
    }

    #[test]
    fn test_column_transformer_column_out_of_bounds() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0], 1, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::Indices(vec![5]));

        let result = ct.fit(&data);
        assert!(matches!(
            result,
            Err(PreprocessingError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_column_transformer_minmax_scaler() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_minmax_scaler(MinMaxScaler::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Check values are in [0, 1] range
        let vals = transformed.ravel().to_vec();
        for &v in &vals {
            assert!(v >= -1e-6 && v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_column_transformer_robust_scaler() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_robust_scaler(RobustScaler::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();
        assert_eq!(transformed.shape(), (3, 2));
    }

    #[test]
    fn test_column_transformer_maxabs_scaler() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_maxabs_scaler(MaxAbsScaler::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Check values are in [-1, 1] range
        let vals = transformed.ravel().to_vec();
        for &v in &vals {
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_column_transformer_normalizer() {
        let data = Tensor2D::<CpuBackend>::new(vec![3.0f32, 4.0, 6.0, 8.0], 2, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_normalizer(Normalizer::new(NormType::L2), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Each row should have unit norm
        let vals = transformed.ravel().to_vec();
        // Row 0: [3, 4] -> [0.6, 0.8], norm = 1
        assert!((vals[0] - 0.6).abs() < 1e-6);
        assert!((vals[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_column_transformer_simple_imputer() {
        let data =
            Tensor2D::<CpuBackend>::new(vec![1.0f32, f32::NAN, 2.0, 10.0, f32::NAN, 20.0], 3, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_simple_imputer(SimpleImputer::new(ImputeStrategy::Mean), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Check no NaN values in output
        let vals = transformed.ravel().to_vec();
        for &v in &vals {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn test_column_transformer_ordinal_encoder() {
        // Categories: 0, 1, 2
        let data = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0], 3, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_ordinal_encoder(OrdinalEncoder::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();
        assert_eq!(transformed.shape(), (3, 2));
    }

    #[test]
    fn test_column_transformer_inverse_transform_not_supported() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0], 2, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        let result = fitted.inverse_transform(&transformed);
        assert!(matches!(
            result,
            Err(PreprocessingError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_column_transformer_empty_data() {
        let data = Tensor2D::<CpuBackend>::zeros(0, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);

        let result = ct.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_transformer_serialization_all_types() {
        // Test serialization with multiple transformer types
        let data = Tensor2D::<CpuBackend>::new(
            vec![1.0f32, 0.0, 10.0, 2.0, 1.0, 20.0, 3.0, 0.0, 30.0],
            3,
            3,
        );

        let pipeline = Pipeline::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new())
            .add_minmax_scaler(MinMaxScaler::new());

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_minmax_scaler(MinMaxScaler::new(), ColumnSpec::Indices(vec![0]))
            .add_one_hot_encoder(OneHotEncoder::new(), ColumnSpec::Indices(vec![1]))
            .add_ordinal_encoder(OrdinalEncoder::new(), ColumnSpec::Indices(vec![2]))
            .add_pipeline(pipeline, ColumnSpec::Indices(vec![0]));

        let fitted = ct.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_column_transformer_all_types.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedColumnTransformer::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());
        assert_eq!(loaded.n_features_out(), fitted.n_features_out());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_column_transformer_step_names() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0], 2, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::Indices(vec![0]))
            .add_minmax_scaler(MinMaxScaler::new(), ColumnSpec::Indices(vec![1]));

        let fitted = ct.fit(&data).unwrap();
        let names = fitted.step_names();

        assert_eq!(names.len(), 2);
        assert_eq!(names[0].0, "StandardScaler");
        assert_eq!(names[1].0, "MinMaxScaler");
    }

    #[test]
    fn test_column_transformer_serialization_comprehensive() {
        // Test serialization with all types that need coverage
        let data = Tensor2D::<CpuBackend>::new(
            vec![1.0f32, f32::NAN, 0.0, 2.0, 20.0, 1.0, 3.0, 30.0, 0.0],
            3,
            3,
        );

        // Pipeline with multiple step types for serialization coverage
        let pipeline = Pipeline::<CpuBackend>::new()
            .add_simple_imputer(SimpleImputer::new(ImputeStrategy::Mean))
            .add_standard_scaler(StandardScaler::new())
            .add_minmax_scaler(MinMaxScaler::new())
            .add_robust_scaler(RobustScaler::new())
            .add_maxabs_scaler(MaxAbsScaler::new())
            .add_normalizer(Normalizer::new(NormType::L2));

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_robust_scaler(RobustScaler::new(), ColumnSpec::Indices(vec![0]))
            .add_maxabs_scaler(MaxAbsScaler::new(), ColumnSpec::Indices(vec![1]))
            .add_normalizer(Normalizer::new(NormType::L2), ColumnSpec::Indices(vec![2]))
            .add_simple_imputer(
                SimpleImputer::new(ImputeStrategy::Mean),
                ColumnSpec::Indices(vec![0]),
            )
            .add_pipeline(pipeline, ColumnSpec::Indices(vec![0, 1]));

        let fitted = ct.fit(&data).unwrap();

        let temp_file = std::env::temp_dir().join("test_column_transformer_comprehensive.bin");
        fitted.save_to_file(&temp_file).unwrap();

        let loaded = FittedColumnTransformer::<CpuBackend>::load_from_file(&temp_file).unwrap();

        assert_eq!(loaded.n_features_in(), fitted.n_features_in());
        assert_eq!(loaded.n_features_out(), fitted.n_features_out());

        // Verify transform gives same results
        let t1 = fitted.transform(&data).unwrap();
        let t2 = loaded.transform(&data).unwrap();

        let v1 = t1.ravel().to_vec();
        let v2 = t2.ravel().to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            if !a.is_nan() && !b.is_nan() {
                assert!((a - b).abs() < 1e-5);
            }
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_column_transformer_fit_transform() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0, 2.0, 20.0], 2, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);

        let transformed = ct.fit_transform(&data).unwrap();
        assert_eq!(transformed.shape(), (2, 2));
    }

    #[test]
    fn test_column_transformer_empty_transformers() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0], 1, 2);

        let ct = ColumnTransformer::<CpuBackend>::new();
        let result = ct.fit(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_column_transformer_duplicate_columns() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0], 1, 2);

        // Add same columns twice
        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::Indices(vec![0]))
            .add_minmax_scaler(MinMaxScaler::new(), ColumnSpec::Indices(vec![0]));

        // Should work but duplicate column 0
        let fitted = ct.fit(&data).unwrap();
        let transformed = fitted.transform(&data).unwrap();

        // Should have 2 output columns (one for each transformer)
        assert_eq!(transformed.shape(), (1, 2));
    }

    #[test]
    fn test_column_transformer_extract_params() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0], 1, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let params = fitted.extract_params();

        assert_eq!(params.n_features_in, 2);
        assert!(!params.steps.is_empty());
    }

    #[test]
    fn test_column_transformer_from_params_roundtrip() {
        let data = Tensor2D::<CpuBackend>::new(vec![1.0f32, 10.0], 1, 2);

        let ct = ColumnTransformer::<CpuBackend>::new()
            .add_standard_scaler(StandardScaler::new(), ColumnSpec::All);

        let fitted = ct.fit(&data).unwrap();
        let params = fitted.extract_params();
        let restored = FittedColumnTransformer::<CpuBackend>::from_params(params).unwrap();

        assert_eq!(restored.n_features_in(), fitted.n_features_in());
        assert_eq!(restored.n_features_out(), fitted.n_features_out());
    }
}
