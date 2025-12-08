pub struct Unfitted;
pub struct Fitted;

pub struct LinearModel<S>{
    weights: Vec<f64>,
    bias: f64,
    lr: f64,
    lambda: f64,
    max_steps: usize,
    delta_converged: f64,
    batch_size: usize,

    _state: std::marker::PhantomData<S>,
}

impl LinearModel<Unfitted>{
    pub fn new(nfeatures: usize) -> Self{
        Self{weights: vec![0.0; nfeatures], 
            bias: 0.0,
            lr: 1e-4,
            lambda: 1.,
            delta_converged: 1e-3,
            batch_size: 64,
            max_steps: 1000, 
            _state: std::marker::PhantomData}
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> LinearModel<Fitted>{
        for _it in 0..self.max_steps{
            let dotproduct = |x: &Vec<f64>| -> f64 {
            let res: f64 = x.iter().zip(self.weights.clone()).map(|(x, y)| {x * y}).sum();
            res + self.bias
            };

            let preds: Vec<_> = x.iter().map(dotproduct).collect();
            let grad_bias: f64 = preds.iter().zip(y.iter()).map(|(p, t)| {2. * (p - t)}).sum::<f64>() / preds.len() as f64;
            self.bias -= self.lr * grad_bias;


            let pred_diff = preds.iter().zip(y.iter()).map(|(p, t)| {2. / preds.len() as f64 * (p - t)});
            let mut wgrad = Vec::with_capacity(self.weights.len());
            for i in 0..x[0].len(){
                let feature: Vec<_> = x.iter().map(|x| {x[i]}).collect();
                let feature_grad: f64 = feature.iter().zip(pred_diff.clone()).map(|(f, g )| { 2. * f * g / preds.len() as f64}).sum();
                wgrad.push(feature_grad);
            }

            wgrad = wgrad.iter().zip(self.weights.iter()).map(|(g, w)| {g + w * 2. * self.lambda}).collect();

            let wgrad_norm: f64 = wgrad.iter().map(|x| {x * x}).sum();
            
            self.weights = self.weights.iter().zip(wgrad.iter()).map(|(w, g)| {w - self.lr * g}).collect();

            if wgrad_norm.sqrt() < self.delta_converged{
                break;
            }

        }

        println!("weights {:?} bias {}", self.weights, self.bias);
        LinearModel::<Fitted>::new(self.weights.clone(), self.bias)

    }


}

impl LinearModel<Fitted>{
    pub fn new(weights: Vec<f64>, bias: f64) -> Self{
        Self{weights, 
            bias,
            lr: 1e-4,
            lambda: 1.,
            delta_converged: 1e-3,
            batch_size: 64,
            max_steps: 1000, 
            _state: std::marker::PhantomData}
    }
    pub fn predict(&self, x: Vec<f64>) -> f64{
        let dotproduct = |x: &Vec<f64>| -> f64 {
            let res: f64 = x.iter().zip(self.weights.clone()).map(|(x, y)| {x * y}).sum();
            res + self.bias
            };

        dotproduct(&x)
    }

    pub fn predict_batch(&self, x: Vec<Vec<f64>>) -> Vec<f64>{
        let dotproduct = |x: &Vec<f64>| -> f64 {
            let res: f64 = x.iter().zip(self.weights.clone()).map(|(x, y)| {x * y}).sum();
            res + self.bias
            };

        let preds: Vec<_> = x.iter().map(dotproduct).collect();
        preds
    }
}


pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_and_predict_identity() {
        // Простейший случай: y = x (один признак, без смещения)
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.0, 3.0];

        let mut model = LinearModel::<Unfitted>::new(1);
        model.lr = 1e-2; // чуть больше скорость для быстрой сходимости
        model.max_steps = 5000;
        model.lambda = 0.01; // отключите регуляризацию!

        let fitted = model.fit(x, y);

        let pred1 = fitted.predict(vec![1.0]);
        let pred2 = fitted.predict(vec![2.5]);

        // Ожидаем приблизительно y ≈ x
        assert!((pred1 - 1.0).abs() < 0.1);
        assert!((pred2 - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_fit_with_bias() {
        // y = 2 * x + 1
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![1.0, 3.0, 5.0];

        let mut model = LinearModel::<Unfitted>::new(1);
        model.lr = 1e-2;
        model.max_steps = 5000;
        model.lambda = 0.0; // отключаем регуляризацию

        let fitted = model.fit(x, y);

        let pred0 = fitted.predict(vec![0.0]);
        let pred1 = fitted.predict(vec![1.0]);
        let pred3 = fitted.predict(vec![3.0]);

        assert!((pred0 - 1.0).abs() < 0.15);
        assert!((pred1 - 3.0).abs() < 0.15);
        assert!((pred3 - 7.0).abs() < 0.2); // экстраполяция
    }

}
