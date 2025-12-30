pub mod backend;
pub mod model;
pub mod optimizer;
pub mod loss;
pub mod trainer;
pub use backend::{Backend, ScalarOps, CpuBackend};

mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::loss::{Loss, MSELoss};
    use crate::optimizer::{Optimizer, SGD};
    use crate::model::linear::{TrainableModel, LinearParams, LinearModel, Unfitted};

    // Вспомогательная функция для создания (n, 1) матрицы из столбца
    fn col_to_tensor2d(col: &[f64]) -> <CpuBackend as Backend>::Tensor2D {
        let n = col.len();
        let mut data = vec![0.0; n];
        data.copy_from_slice(col);
        (data, n, 1)
    }

    fn slice_to_tensor1d(slice: &[f64]) -> <CpuBackend as Backend>::Tensor1D {
        slice.to_vec()
    }

    #[test]
    fn test_linear_regression_identity() {
        // y = x
        let x_data = &[1.0, 2.0, 3.0, 4.0];
        let y_data = &[1.0, 2.0, 3.0, 4.0];

        let x_tensor = col_to_tensor2d(x_data);
        let y_tensor = slice_to_tensor1d(y_data);

        let device = CpuBackend::default_device();
        let params = LinearParams {
            weights: CpuBackend::zeros_1d(1, &device),
            bias: <CpuBackend as backend::Backend>::Scalar::from_f64(0.0),
        };

        let mut model = LinearModel::<CpuBackend, Unfitted>::new(params);
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
    let w = CpuBackend::get_1d(&model.params().weights, 0);
    let b = model.params().bias;
    println!("Epoch {}: loss={:.6}, w={:.4}, b={:.4}, pred={:?}", epoch, 
             <CpuBackend as backend::Backend>::Scalar::to_f64(loss_val), w, <CpuBackend as backend::Backend>::Scalar::to_f64(b),
            pred);
}
        }

        let fitted = model.into_fitted();
        let pred = fitted.predict(&[2.5]);
        assert!(
            (pred - 2.5).abs() < 0.1,
            "Expected ~2.5, got {}",
            pred
        );
    }

    #[test]
    fn test_linear_regression_with_bias() {
        // y = 2*x + 1
        let x_data = &[0.0, 1.0, 2.0, 3.0];
        let y_data = &[1.0, 3.0, 5.0, 7.0];

        let x_tensor = col_to_tensor2d(x_data);
        let y_tensor = slice_to_tensor1d(y_data);

        let device = CpuBackend::default_device();
        let params = LinearParams {
            weights: CpuBackend::zeros_1d(1, &device),
            bias: <CpuBackend as backend::Backend>::Scalar::from_f64(0.0),
        };

        let mut model = LinearModel::<CpuBackend, Unfitted>::new(params);

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
        let p0 = fitted.predict(&[0.0]);
        let p1 = fitted.predict(&[1.0]);
        let p3 = fitted.predict(&[3.0]);

        assert!((p0 - 1.0).abs() < 0.2, "p0 = {}", p0);
        assert!((p1 - 3.0).abs() < 0.2, "p1 = {}", p1);
        assert!((p3 - 7.0).abs() < 0.3, "p3 = {}", p3);
    }
}
