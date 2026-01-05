# 0003-semantic-axis-labeling

## Status

Proposed

## Context

In machine learning, 2D tensors typically follow a semantic convention: **rows represent samples**, and **columns represent features**. Accidentally transposing such a matrix — for example, feeding a `[features × batch]` tensor where `[batch × features]` is expected — leads to silent, catastrophic errors: gradients flow through the wrong dimensions, and models learn nonsense.

While shape checks can catch some mismatches at runtime, they cannot prevent logical confusion at the API level. We aim to leverage Rust’s type system to **encode semantic axis roles** (e.g., `Samples`, `Features`, `Classes`) directly in tensor types, making misuse a compile-time error.

This aligns with ADR-0001: just as we separate concerns like loss and optimizer, we also seek to separate **data semantics** from raw storage.

## Decision

Introduce **semantic axis labels** via phantom types for high-level ML data structures:

- Define unit structs: `pub struct Samples;`, `pub struct Features;`, `pub struct Classes;`, etc.
- Introduce a typed alias:  
  `pub type Dataset<B> = Matrix<B, Samples, Features>;`  
  `pub type WeightMatrix<B> = Matrix<B, Features, Hidden>;`
- Keep low-level `Tensor2D<B>` unchanged for backend operations (BLAS, allocation, etc.)
- Provide explicit, fallible conversions from `Tensor2D<B>` to `Dataset<B>` that validate shape expectations
- ML algorithms (e.g., `LinearModel::fit`) accept `Dataset<B>`, not raw tensors

This approach embeds **domain semantics** into the type system without sacrificing performance or backend flexibility.

## Consequences

✅ Prevents sample/feature transposition bugs at compile time  
✅ Self-documenting APIs: function signatures encode data layout intent  
✅ Complements ADR-0001’s separation of concerns—training logic now operates on semantically labeled data  
✅ Zero runtime cost (phantom types + compile-time checks)  
⚠️ Adds minor verbosity in type signatures  
⚠️ Requires explicit conversion from raw tensors (a feature, not a bug)

## Alternatives Considered

- **Runtime-only shape assertions**  
  Prone to late failure; doesn’t prevent logical misuse in composition.

- **Full named-axis system (like xarray)**  
  Overkill for typical batched ML workloads; unnecessary complexity.

- **Do nothing**  
  Accept transposition bugs as a tax — rejected due to safety goals.

## References

- ADR-0001: Separate trainer, losses, and optimizers
- Inspired by dimensional typing in scientific Rust crates and array semantics in `ndarray`/`xarray`