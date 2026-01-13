# machinelearne_rs

> A type-safe, zero-overhead machine learning library in Rust — built for learners, by learners.

`machinelearne_rs` is a minimal, composable ML library designed around **explicit separation of concerns**:  
- **Models** know how to predict and compute gradients.  
- **Losses** know how to score predictions and emit per-sample gradients.  
- **Optimizers** update parameters based on gradients.  
- **Regularizers** penalize model complexity independently.  
- **Trainers** orchestrate the loop — nothing more.

No hidden state. No dynamic dispatch. No runtime surprises. Just pure, generic Rust.

Inspired by functional design and numerical rigor — not by auto-generated boilerplate.

---

## 🧠 Core Principles

- ✅ **Type safety first**: Leverage Rust’s type system to prevent shape mismatches and invalid training loops at compile time.
- ✅ **Separation of concerns**: Following [RFC 0001: Separate Trainer, Losses, and Regularizers](./docs/0001-separate-trainer-losses.md).
- ✅ **Backend agnostic**: CPU-first, but pluggable (e.g., future GPU support via traits).
- ✅ **No magic**: What you write is what you get — no implicit graph building or lazy evaluation.

---

## 🚀 Quick Example

Train a linear regressor with MSE loss and SGD:

```rust
use machinelearne_rs::{
    CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{linear::LinearRegressor, InferenceModel},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

fn main() {
    let model = LinearRegressor::new(2); // 2 features
    let loss = MSELoss;
    let opt = SGD::new(0.01);
    let reg = NoRegularizer;

    let trainer = Trainer::builder(loss, opt, reg)
        .batch_size(4)
        .max_epochs(1000)
        .build();

    let x = vec![
        vec![1.0, 1.0],
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 2.0],
    ];
    let y = vec![3.0, 4.0, 5.0, 6.0]; // y = x₀ + 2*x₁

    let dataset = InMemoryDataset::new(x, y).unwrap();
    let fitted = trainer.fit(model, &dataset).unwrap();

    let input = Tensor1D::<CpuBackend>::new(vec![3.0, 4.0]);
    let pred = fitted.predict(&input);
    println!("Prediction: {:.2}", pred.data()[0]); // ≈ 11.0
}

## 📦 Current Features

- **Models**: `LinearRegressor`
- **Losses**: `MSELoss`
- **Optimizers**: `SGD`
- **Regularizers**: `NoRegularizer` (L2 planned)
- **Backends**: `CpuBackend` (dense `f32` tensors)
- **Datasets**: In-memory only (`InMemoryDataset`)

> This is a **learning-focused prototype**, not a production framework. Contributions and experiments welcome!