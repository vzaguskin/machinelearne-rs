use std::ops::Range;
use crate::backend::{Backend, Tensor1D, Tensor2D};
use crate::dataset::Dataset;
#[derive (Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_in_memory_dataset_new_success() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_ok());
    }

    #[test]
    fn test_in_memory_dataset_new_mismatched_lengths() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_err());
        assert_eq!(dataset.unwrap_err(), "x and y must have same length");
    }

    #[test]
    fn test_in_memory_dataset_new_empty() {
        let x = vec![];
        let y = vec![];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_err());
        assert_eq!(dataset.unwrap_err(), "Dataset is empty");
    }

    #[test]
    fn test_in_memory_dataset_new_uneven_rows() {
        let x = vec![vec![1.0, 2.0], vec![3.0]]; // вторая строка короче
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y);
        assert!(dataset.is_err());
        assert_eq!(
            dataset.unwrap_err(),
            "All rows must have the same number of features"
        );
    }

    #[test]
    fn test_in_memory_dataset_len() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();
        assert_eq!(dataset.len(), Some(2));
    }

    #[test]
    fn test_in_memory_dataset_batches_integration() {
        let x = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let y = vec![1.0, 0.0, 1.0];
        let dataset = InMemoryDataset::new(x, y).unwrap();

        let mut batches = dataset.batches::<CpuBackend>(2);
        let batch1 = batches.next().unwrap().unwrap();
        assert_eq!(batch1.0.shape(), (2, 2));
        assert_eq!(batch1.1.to_vec(), vec![1.0, 0.0]);

        let batch2 = batches.next().unwrap().unwrap();
        assert_eq!(batch2.0.shape(), (1, 2));
        assert_eq!(batch2.1.to_vec(), vec![1.0]);

        assert!(batches.next().is_none());
    }
}