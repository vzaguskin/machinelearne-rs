pub trait Loss<B: Backend, M: Model>{
    type Target;
    pub fn grad(params: M::Params, predictions: M::Prediction, target: Target);
}


struct MSELoss;

impl Loss for MSELoss<B, LinearModel<B, Unfitted>>{
    type Target = B::Tensor1D;

    fn grad(params: M::Params, predictions: M::Prediction, target: Target) -> M::Params{

        let diffs = B::minus_vec_vec(&predictions, &target);

        // --- Bias gradient: 2/n * sum(diffs) ---
        let n_samples_f = B::shape_2d(&predictions).0;
        let grad_bias = (2.0 / n_samples_f) * B::sum_1d(&diffs).to_f64();
        bias = bias - lr * B::scalar(grad_bias, device);

        // --- Weights gradient: 2/n * X^T @ diffs + 2*lambda*weights ---
        let w_grad = B::matvec(&B::transpose(&x), &diffs);
        let scale = B::scalar(2.0 / n_samples_f, device);
        let w_grad = B::scale_1d(scale, &w_grad);
        let w_reg = B::scale_1d(lambda * B::scalar(2., device), &weights);

        let w_grad = B::plus_vec_vec(&w_grad, &w_reg);

    }
}