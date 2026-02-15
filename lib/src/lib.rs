//! # rust-ml
//!
//! A type-safe machine learning library in Rust with pluggable backends and strict
//! separation between training and inference phases.
//!
//! ## Core Design Principles
//!
//! - **Stateful Type Safety**: Models carry their training state in the type system
//!   (`Unfitted` vs `Fitted`), preventing invalid operations at compile time.
//! - **Training/Inference Separation**: Trained models contain only prediction parameters;
//!   training logic lives in separate components (losses, optimizers, trainers).
//! - **Backend Agnosticism**: Abstract `Backend` trait enables CPU/GPU implementations
//!   without changing model code.
//! - **Zero-Cost Abstractions**: Generics and traits provide flexibility without runtime overhead.
//!
//! ## Quick Start
//!
//! ```rust
//! use machinelearne_rs::backend::CpuBackend;
//! use machinelearne_rs::model::linear::{LinearModel, Unfitted};
//! use machinelearne_rs::loss::MSELoss;
//! use machinelearne_rs::optimizer::SGD;
//!
//! // Create an untrained linear model (1 input feature)
//! let mut model = LinearModel::<CpuBackend, Unfitted>::new(1);
//!
//! // Training loop (simplified)
//! // for epoch in 0..100 {
//! //     let pred = model.forward(&x_tensor);
//! //     let grad = MSELoss::grad_wrt_prediction(&pred, &y_tensor);
//! //     let grads = model.backward(&x_tensor, &grad);
//! //     let new_params = SGD::new(0.01).step(model.params(), &grads);
//! //     model.update_params(&new_params);
//! // }
//!
//! // Convert to inference-optimized model
//! // let fitted = model.into_fitted();
//! // let prediction = fitted.predict(&input_tensor);
//! ```
//!
//! ## Module Structure
//!
//! - `backend` — Tensor abstractions and computation primitives (`Tensor1D`, `Tensor2D`)
//! - `model` — ML model implementations with stateful type parameters
//! - `loss` — Differentiable loss functions (MSE, CrossEntropy, etc.)
//! - `optimizer` — Parameter update algorithms (SGD, Adam)
//! - `trainer` — High-level training loop orchestration
//! - `regularizers` — Weight regularization strategies (L1, L2)
//! - `dataset` — Data loading and preprocessing utilities
//! - `serialization` — Model persistence formats
//!
//! ## Example Projects
//!
//! See the `examples/` directory for complete training pipelines demonstrating:
//! - Linear regression with SGD
//! - Regularization techniques
//! - Custom backend integration
//! - Model serialization workflows

pub mod backend;

/// Data loading utilities and dataset abstractions.
pub mod dataset;

/// Data preprocessing transformers for ML pipelines.
pub mod preprocessing;

/// Differentiable loss functions for model training.
pub mod loss;

/// Machine learning models with compile-time state safety.
pub mod model;

/// Optimization algorithms for parameter updates.
pub mod optimizer;

/// Weight regularization strategies to prevent overfitting.
pub mod regularizers;

/// Model persistence and format conversion utilities.
pub mod serialization;

/// High-level training loop orchestration.
pub mod trainer;

/// Re-export of core backend types for convenient usage.
pub use backend::{Backend, CpuBackend, ScalarOps, Tensor1D, Tensor2D};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::loss::{Loss, MSELoss};
    use crate::model::linear::{InferenceModel, LinearModel, TrainableModel, Unfitted};
    use crate::optimizer::{Optimizer, SGD};

    // Helper function to create (n, 1) matrix from column data
    fn col_to_tensor2d<B: Backend>(col: &[f32]) -> Tensor2D<B> {
        let n = col.len();
        let mut data = vec![0.0; n];
        data.copy_from_slice(col);
        Tensor2D::<B>::new(data, n, 1)
    }

    fn slice_to_tensor1d<B: Backend>(slice: &[f32]) -> Tensor1D<B> {
        Tensor1D::<B>::new(slice.to_vec())
    }

    #[test]
    fn test_linear_regression_identity() {
        // y = x
        let x_data: &[f32; 4] = &[1.0, 2.0, 3.0, 4.0];
        let y_data: &[f32; 4] = &[1.0, 2.0, 3.0, 4.0];

        let x_tensor = col_to_tensor2d(x_data);
        let y_tensor = slice_to_tensor1d(y_data);

        let mut model = LinearModel::<CpuBackend, Unfitted>::new(1);
        let loss_fn = MSELoss;
        let optimizer = SGD::new(0.01);

        for epoch in 0..200 {
            let pred = model.forward(&x_tensor);
            let grad_pred = Loss::<CpuBackend>::grad_wrt_prediction(&loss_fn, &pred, &y_tensor);
            let grads = model.backward(&x_tensor, &grad_pred);
            let new_params = optimizer.step(&model.params(), &grads);
            model.update_params(&new_params);
            if epoch % 5 == 0 {
                let loss_val = Loss::<CpuBackend>::loss(&loss_fn, &pred, &y_tensor);
                let w = &model.params().weights.to_vec()[0];
                let b = model.params().bias.data.to_f64();
                println!(
                    "Epoch {}: loss={:.6}, w={:.4}, b={:.4}, pred={:?}",
                    epoch,
                    loss_val.data.to_f64(),
                    w,
                    b,
                    pred.to_vec()
                );
            }
        }

        let fitted = model.into_fitted();
        let inp = slice_to_tensor1d::<CpuBackend>(&[2.5]);
        let pred = fitted.predict(&inp).data.to_f64();
        assert!((pred - 2.5).abs() < 0.1, "Expected ~2.5, got {}", pred);
    }

    #[test]
    fn test_linear_regression_with_bias() {
        // y = 2*x + 1
        let x_data = &[0.0, 1.0, 2.0, 3.0];
        let y_data = &[1.0, 3.0, 5.0, 7.0];

        let x_tensor = col_to_tensor2d(x_data);
        let y_tensor = slice_to_tensor1d(y_data);

        let mut model = LinearModel::<CpuBackend, Unfitted>::new(1);

        let loss_fn = MSELoss;
        let optimizer = SGD::new(0.01);

        for _ in 0..3000 {
            let pred = model.forward(&x_tensor);
            let grad_pred = Loss::<CpuBackend>::grad_wrt_prediction(&loss_fn, &pred, &y_tensor);
            let grads = model.backward(&x_tensor, &grad_pred);
            let new_params = optimizer.step(model.params(), &grads);
            model.update_params(&new_params);
        }

        let fitted = model.into_fitted();
        let p0 = fitted.predict(&slice_to_tensor1d::<CpuBackend>(&[0.0]));
        let p1 = fitted.predict(&slice_to_tensor1d::<CpuBackend>(&[1.0]));
        let p3 = fitted.predict(&slice_to_tensor1d::<CpuBackend>(&[3.0]));

        assert!(
            (p0.data.to_f64() - 1.0).abs() < 0.2,
            "p0 = {}",
            p0.data.to_f64()
        );
        assert!(
            (p1.data.to_f64() - 3.0).abs() < 0.2,
            "p1 = {}",
            p1.data.to_f64()
        );
        assert!(
            (p3.data.to_f64() - 7.0).abs() < 0.3,
            "p3 = {}",
            p3.data.to_f64()
        );
    }
}
