/// A marker type indicating that a model is **not yet trained**.
///
/// This phantom type is used in generic parameters (e.g., `LinearRegressor<Unfitted>`)
/// to enforce compile-time guarantees:
/// - Training methods (like `Trainer::fit`) require an `Unfitted` model.
/// - Inference methods (`predict`) are **not available** until the model is converted to `Fitted`.
///
/// This prevents accidental use of an untrained model for prediction.
pub struct Unfitted;

/// A marker type indicating that a model has been **fully trained**.
///
/// After training, a model is converted from `Model<Unfitted>` to `Model<Fitted>`,
/// which implements [`InferenceModel`] and can be serialized or used for prediction.
///
/// A `Fitted` model contains **only inference parameters** — no optimizer state,
/// loss function, or training hyperparameters (per ADR-0001).
pub struct Fitted;