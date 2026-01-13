pub mod backend;

pub mod model;
pub mod optimizer;
pub mod loss;
pub mod trainer;
pub mod regularizers;
pub mod dataset;
pub use backend::{Backend, ScalarOps, CpuBackend, Tensor1D, Tensor2D};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::loss::{Loss, MSELoss};
    use crate::optimizer::{Optimizer, SGD};
    use crate::model::linear::{TrainableModel, LinearModel, Unfitted, InferenceModel};

    // Вспомогательная функция для создания (n, 1) матрицы из столбца
    fn col_to_tensor2d<B: Backend>(col: &[f32]) -> Tensor2D<B> {
        let n = col.len();
        let mut data = vec![0.0; n];
        data.copy_from_slice(col);
        Tensor2D::<B>::new(data, n, 1)
    }

    fn slice_to_tensor1d<B: Backend>(slice: &[f32]) -> Tensor1D<B>  {
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
            println!("Epoch {}: loss={:.6}, w={:.4}, b={:.4}, pred={:?}", epoch, 
                    loss_val.data.to_f64(), w, b,
                    pred.to_vec());
}
        }

        let fitted = model.into_fitted();
        let inp = slice_to_tensor1d::<CpuBackend>(&[2.5]);
        let pred = fitted.predict(&inp).data.to_f64();
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

        assert!((p0.data.to_f64() - 1.0).abs() < 0.2, "p0 = {}", p0.data.to_f64());
        assert!((p1.data.to_f64() - 3.0).abs() < 0.2, "p1 = {}", p1.data.to_f64());
        assert!((p3.data.to_f64() - 7.0).abs() < 0.3, "p3 = {}", p3.data.to_f64());
    }
}
