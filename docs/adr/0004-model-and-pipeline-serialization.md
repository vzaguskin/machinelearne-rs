# 0004-serializable-fitted-models.md

## Title  
Serializable Fitted Models via Data-Centric Parameter Representation

## Status  
Proposed

## Context  
After training, a model enters the `Fitted` state and should be deployable, versioned, and persisted. However, the internal parameter representation (e.g., `Tensor1D<B>`, `Scalar<B>`) is tied to a specific backend and may contain non-serializable resources (e.g., GPU buffers). Directly deriving `serde::Serialize` on models is unsafe and couples the core interface to a serialization format.

We need a **safe, backend-agnostic, and format-agnostic** way to serialize and deserialize **only the numerical parameters** of a fitted model.

## Decision  
Introduce a new trait:

```rust
pub trait SerializableParams {
    type Error: std::error::Error + Send + Sync + 'static;
    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error>;
    fn from_bytes(&[u8]) -> Result<Self, Self::Error>;
}

And extend InferenceModel (which represents the Fitted state) with:

```rust
pub trait InferenceModel<B: Backend>: Sized {
    type ParamsRepr: SerializableParams;
    // ... existing associated types ...

    fn extract_params(&self) -> Self::ParamsRepr;
    fn from_params(params: Self::ParamsRepr) -> Result<Self, Box<dyn std::error::Error>>;

    // Optional convenience methods
    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()>;
    fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>>;
}

Provide an optional serde integration via Cargo feature:

```toml
[features]
serde = ["dep:serde", "dep:bincode"]

When enabled, any `#[derive(Serialize, Deserialize)]` struct automatically implements `SerializableParams` using `bincode`.

Concrete models (e.g., `LinearModel<B, Fitted>`) must:
- Define a **plain-data** `ParamsRepr` (e.g., `SerializableLinearParams { weights: Vec<f32>, bias: f32 }`),
- Implement conversion to/from their internal `Params` type.

This ensures:
- **No training state** is saved (consistent with `0001-separate-trainer-losses.md`),
- **Backend independence** of serialized artifacts,
- **Format flexibility** (binary, JSON, MessagePack, etc.).

## Consequences  
✅ Fitted models can be safely persisted and loaded across processes/machines.  
✅ Serialized format is human-inspectable (if using JSON) or compact (if using bincode).  
✅ No coupling between core model logic and serialization libraries.  
⚠️ Requires implementers to provide a data-centric `ParamsRepr`.  
⚠️ Adds one more associated type to `InferenceModel`.

## Alternatives Considered  
1. **Derive `Serialize` directly on models**  
   → Rejected: ties models to `serde`, unsafe for GPU backends.

2. **Use `bincode::serialize` on internal `Params`**  
   → Rejected: `Tensor1D<B>` is not serializable by design.

3. **Postpone until pipelines are designed**  
   → Rejected: serialization is a prerequisite for pipeline persistence.

## References  
- [0001-separate-trainer-losses.md](0001-separate-trainer-losses.md): “Fitted model is free from training hyperparameters”  
- [Serde ecosystem](https://serde.rs/)  
- [Bincode](https://github.com/bincode-org/bincode)