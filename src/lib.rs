pub mod backend;
pub mod model;
pub use backend::{Backend, ScalarOps, CpuBackend};


pub struct Unfitted;
pub struct Fitted;

/// Linear regression model parametrized by backend and state.
pub struct LinearModel<B: Backend, S> {
    weights: B::Tensor1D,
    bias: B::Scalar,
    lr: B::Scalar,
    lambda: B::Scalar,
    max_steps: usize,
    delta_converged: B::Scalar,
    batch_size: usize,

    _state: std::marker::PhantomData<S>,
}

// === Unfitted state ===
impl<B: Backend> LinearModel<B, Unfitted> {
    /// Create a new unfitted linear model.
    pub fn new(n_features: usize, device: &B::Device) -> Self {
        Self {
            weights: B::zeros_1d(n_features, device),
            bias: B::scalar(0.0, device),
            lr: B::scalar(1e-2, device),
            lambda: B::scalar(0.1, device),
            delta_converged: B::scalar(1e-3, device),
            batch_size: 64,
            max_steps: 5000,
            _state: std::marker::PhantomData,
        }
    }

    /// Fit the model using full-batch gradient descent.
    pub fn fit(self, x: B::Tensor2D, y: B::Tensor1D, device: &B::Device) -> LinearModel<B, Fitted> {
        let mut weights = self.weights;
        let mut bias = self.bias;
        let n_samples = B::shape_2d(&x).0;
        let n_features = B::len_1d(&weights);

        assert_eq!(n_samples, B::len_1d(&y), "X rows must match Y length");

        let n_samples_f = n_samples as f64;
        let lr = self.lr;
        let lambda = self.lambda;
        println!("Starting fit with max_steps = {}", self.max_steps);
        for _step in 0..self.max_steps {
            // --- Forward pass: preds = X @ weights + bias ---
            let mut preds = B::matvec(&x, &weights);
            B::add_scalar_1d_inplace(&mut preds, bias);

            let diffs = B::minus_vec_vec(&preds, &y);

            // --- Bias gradient: 2/n * sum(diffs) ---
            let grad_bias = (2.0 / n_samples_f) * B::sum_1d(&diffs).to_f64();
            bias = bias - lr * B::scalar(grad_bias, device);

            // --- Weights gradient: 2/n * X^T @ diffs + 2*lambda*weights ---
            let w_grad = B::matvec(&B::transpose(&x), &diffs);
            let scale = B::scalar(2.0 / n_samples_f, device);
            let w_grad = B::scale_1d(scale, &w_grad);
            let w_reg = B::scale_1d(lambda * B::scalar(2., device), &weights);
  
            let w_grad = B::plus_vec_vec(&w_grad, &w_reg);
            for j in 0..n_features {
                let w_old = B::get_1d(&weights, j);
                let g = B::get_1d(&w_grad, j);
                B::set_1d(&mut weights, j, w_old - lr * g);
            }

            // --- Convergence check ---
            let grad_norm_sq: f64 = (0..n_features).map(|j| {
                let g = B::get_1d(&w_grad, j).to_f64();
                g * g
            }).sum();
            if grad_norm_sq.sqrt() < self.delta_converged.to_f64() {
                break;
            }
            //println!("{_step}, {:?} {:?}", B::get_1d(&weights, 0).to_f64(), bias.to_f64());
        }

        LinearModel {
            weights,
            bias,
            lr: self.lr,
            lambda: self.lambda,
            max_steps: self.max_steps,
            delta_converged: self.delta_converged,
            batch_size: self.batch_size,
            _state: std::marker::PhantomData,
        }
    }
}

// === Fitted state ===
impl<B: Backend> LinearModel<B, Fitted> {
    /// Predict on a single sample.
    pub fn predict(&self, x: B::Tensor1D) -> B::Scalar {
        B::dot(&x, &self.weights) + self.bias
    }

    /// Predict on a batch.
    pub fn predict_batch(&self, x: B::Tensor2D) -> B::Tensor1D {
        let mut preds = B::matvec(&x, &self.weights);
        let n = B::shape_2d(&x).0;
        for i in 0..n {
            let p = B::get_1d(&preds, i);
            B::set_1d(&mut preds, i, p + self.bias);
        }
        preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use backend::CpuBackend;
    use crate::backend::Device;
    pub use model::LinearModel;

    #[test]
    fn test_new_structure() {
    }

    #[test]
    fn test_fit_predict_identity() {
        let device = Device::Cpu;
        // X: 3 samples, 1 feature → flat buffer: [1.0, 2.0, 3.0], shape (3,1)
        let x = (vec![1.0, 2.0, 3.0], 3, 1);
        let y = vec![1.0, 2.0, 3.0];

        let model = LinearModel::<CpuBackend, Unfitted>::new(1, &device);
        let fitted = model.fit(x, y, &device);

        let pred = fitted.predict(vec![2.5]); // → f64 directly
        assert!((pred - 2.5).abs() < 0.2);
    }

    #[test]
    fn test_fit_with_bias() {
        // y = 2 * x + 1
        let device = Device::Cpu;
        let x = (vec![0.0, 1.0, 2.0], 3, 1);
        let y = vec![1.0, 3.0, 5.0];

        let model = LinearModel::<CpuBackend, Unfitted>::new(1, &device);
        //model.lr = 1e-2;
        //model.max_steps = 5000;
        //model.lambda = 0.0; // отключаем регуляризацию

        let fitted = model.fit(x, y, &device);

        let pred0 = fitted.predict(vec![0.0]);
        let pred1 = fitted.predict(vec![1.0]);
        let pred3 = fitted.predict(vec![3.0]);

        assert!((pred0 - 1.0).abs() < 0.15);
        assert!((pred1 - 3.0).abs() < 0.15);
        assert!((pred3 - 7.0).abs() < 0.2); // экстраполяция
    }
}
