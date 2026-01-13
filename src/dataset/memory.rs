use std::ops::Range;
use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::dataset::Dataset;
pub struct InMemoryDataset {
    x: Vec<Vec<f32>>,
    y: Vec<f32>,
}

impl InMemoryDataset {
    pub fn new(x: Vec<Vec<f32>>, y: Vec<f32>) -> Result<Self, String> {
        if x.len() != y.len() {
            return Err("x and y must have same length".into());
        }
        if x.is_empty() {
            return Err("Dataset is empty".into());
        }
        let n_features = x[0].len();
        if !x.iter().all(|row| row.len() == n_features) {
            return Err("All rows must have the same number of features".into());
        }
        Ok(Self { x, y })
    }
}

impl Dataset for InMemoryDataset {
    type Error = std::convert::Infallible;
    type Item = (Vec<f32>, f32);

    fn len(&self) -> Option<usize> {
        Some(self.x.len())
    }

    fn get_batch<B: Backend>(
        &self,
        range: Range<usize>,
    ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> {
        let batch_x = &self.x[range.clone()];
        let batch_y = &self.y[range];

        let batch_size = batch_x.len();
        let n_features = batch_x[0].len();

        let data = batch_x.iter().flat_map(|row| row.iter()).copied().collect();
        let x_tensor = Tensor2D::<B>::new(data, batch_size, n_features);

        let y_tensor = Tensor1D::<B>::new(batch_y.to_vec());

        Ok((x_tensor, y_tensor))
    }
}