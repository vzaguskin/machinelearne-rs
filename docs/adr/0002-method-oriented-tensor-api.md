# 2. method-oriented-tensor-api

## Status

Proposed

## Context

The current tensor API uses a purely functional style with free functions on a `Backend` trait:  
`Backend::sub_1d(&x, &y)`, `Backend::matmul(&a, &b)`, etc. While this is composable and explicit, it leads to verbose, hard-to-chain expressions and reduced ergonomics for users writing numerical code.

Rust’s type system and zero-cost abstractions make it possible to provide a fluent, method-based interface without sacrificing performance or generality. The goal is not to emulate OOP, but to enable **expression-oriented**, **composable**, and **type-safe** numerical code—similar in spirit to what one might write in a functional language with rich type-level structure (e.g., Haskell with `linear` or `accelerate`), but expressed idiomatically in Rust.

We want users to write:

```rust
let logits = weights.matmul(&x).add(&bias);
let loss = logits.sigmoid().bce_with_logits(&targets);

—without runtime overhead, dynamic dispatch, or loss of backend generality (CPU, CUDA, etc.).

## Decision

Introduce strongly-typed tensor wrappers (`Tensor1D<B>`, `Tensor2D<B>`, etc.) that:

- Store backend-specific data and a `PhantomData<B>`
- Implement methods (`sub`, `add`, `matmul`, `sigmoid`, etc.) that delegate to `B: Backend`
- Support method chaining and compositional pipelines
- Remain fully generic over the backend, with all dispatch resolved at compile time

Replace the `enum Tensor<B>` with distinct, rank-encoded types to:

- Eliminate runtime branching on tensor rank
- Enforce shape compatibility statically where possible
- Enable future extensions (e.g., const generics for compile-time shapes)

Implement operator traits (`Add`, `Sub`, `Mul`) optionally, preferring explicit method names for clarity in ML contexts (e.g., `matmul` vs `*`).

## Consequences

✅ Enables fluent, readable, and composable numerical code  
✅ Maintains zero-cost abstraction — no heap allocation, no dynamic dispatch  
✅ Preserves backend agnosticism (CPU, GPU, etc.)  
✅ Aligns with Rust’s ownership and type discipline  
✅ Encourages functional-style pipelines (`x.f().g().h()`) over mutable state  

⚠️ Requires boilerplate for each rank–operation pair (mitigatable via macros)  
⚠️ Increases API surface slightly, but improves usability significantly

## Alternatives Considered

**Keep functional `Backend::op(x, y)` style**  
→ Verbose, hard to read, poor composability. Rejected for usability.

**Use `enum Tensor<B>` with dynamic rank**  
→ Sacrifices type safety and performance. Conflicts with goal of compile-time guarantees.

**Full AD / computational graph (à la `tch` or `candle`)**  
→ Overkill for current scope; deferred to future stages per ADR-0001.

## References

- [ADR-0001: Separate trainer, losses, and optimizers](0001-separate-trainer-losses.md)
- Inspired by design principles of `ndarray`, `candle`, and functional array languages