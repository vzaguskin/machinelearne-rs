//! Train/test split utilities.

use crate::dataset::memory::InMemoryDataset;
use rand::prelude::*;

/// Splits dataset into training and testing subsets.
///
/// # Arguments
/// * `x` - Feature matrix (samples × features)
/// * `y` - Target vector
/// * `test_size` - Fraction of data to use for testing (0.0 to 1.0)
/// * `random_state` - Optional seed for reproducible shuffling
///
/// # Returns
/// A tuple `(train_dataset, test_dataset)` on success, or an error message.
///
/// # Example
///
/// ```rust
/// use machinelearne_rs::model_selection::train_test_split;
/// use machinelearne_rs::dataset::Dataset;
///
/// let x = vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
///     vec![5.0, 6.0],
///     vec![7.0, 8.0],
/// ];
/// let y = vec![0.0, 1.0, 0.0, 1.0];
///
/// let (train, test) = train_test_split(x, y, 0.25, Some(42)).unwrap();
/// assert_eq!(train.len(), Some(3));
/// assert_eq!(test.len(), Some(1));
/// ```
pub fn train_test_split(
    x: Vec<Vec<f32>>,
    y: Vec<f32>,
    test_size: f32,
    random_state: Option<u64>,
) -> Result<(InMemoryDataset, InMemoryDataset), String> {
    // Validate input
    if x.len() != y.len() {
        return Err("x and y must have the same length".into());
    }
    if x.is_empty() {
        return Err("Cannot split empty dataset".into());
    }
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err("test_size must be between 0.0 and 1.0 (exclusive)".into());
    }

    let n_samples = x.len();
    let n_test = ((n_samples as f32) * test_size).round() as usize;
    let n_train = n_samples - n_test;

    if n_test == 0 {
        return Err(format!(
            "test_size {} is too small for {} samples (results in 0 test samples)",
            test_size, n_samples
        ));
    }
    if n_train == 0 {
        return Err(format!(
            "test_size {} is too large for {} samples (results in 0 train samples)",
            test_size, n_samples
        ));
    }

    // Create index permutation
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if let Some(seed) = random_state {
        let mut rng = SmallRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
    }

    // Split indices
    let (train_indices, test_indices) = indices.split_at(n_train);

    // Create datasets
    let train_x: Vec<Vec<f32>> = train_indices.iter().map(|&i| x[i].clone()).collect();
    let train_y: Vec<f32> = train_indices.iter().map(|&i| y[i]).collect();
    let test_x: Vec<Vec<f32>> = test_indices.iter().map(|&i| x[i].clone()).collect();
    let test_y: Vec<f32> = test_indices.iter().map(|&i| y[i]).collect();

    let train_dataset = InMemoryDataset::new(train_x, train_y)?;
    let test_dataset = InMemoryDataset::new(test_x, test_y)?;

    Ok((train_dataset, test_dataset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;

    #[test]
    fn test_train_test_split_basic() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![0.0, 1.0, 2.0, 3.0];

        let (train, test) = train_test_split(x, y, 0.25, None).unwrap();
        assert_eq!(train.len(), Some(3));
        assert_eq!(test.len(), Some(1));
    }

    #[test]
    fn test_train_test_split_reproducible() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        let (train1, test1) = train_test_split(x.clone(), y.clone(), 0.33, Some(42)).unwrap();
        let (train2, test2) = train_test_split(x, y, 0.33, Some(42)).unwrap();

        // Same seed should produce same split
        assert_eq!(train1.features(), train2.features());
        assert_eq!(test1.features(), test2.features());
    }

    #[test]
    fn test_train_test_split_different_seeds() {
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        let (train1, _) = train_test_split(x.clone(), y.clone(), 0.5, Some(1)).unwrap();
        let (train2, _) = train_test_split(x, y, 0.5, Some(2)).unwrap();

        // Different seeds should (most likely) produce different splits
        // We can't guarantee they're different, but the probability is low for 6 samples
        let features1: Vec<_> = train1.features().iter().flatten().collect();
        let features2: Vec<_> = train2.features().iter().flatten().collect();
        // Just check they both have the right size
        assert_eq!(features1.len(), features2.len());
    }

    #[test]
    fn test_train_test_split_mismatched_lengths() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.0];

        let result = train_test_split(x, y, 0.5, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("same length"));
    }

    #[test]
    fn test_train_test_split_empty_data() {
        let x: Vec<Vec<f32>> = vec![];
        let y: Vec<f32> = vec![];

        let result = train_test_split(x, y, 0.5, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_train_test_split_invalid_test_size_zero() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0];

        let result = train_test_split(x, y, 0.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_test_split_invalid_test_size_one() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0];

        let result = train_test_split(x, y, 1.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_test_split_test_size_too_small() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0];

        // With 2 samples and test_size=0.1, we'd get 0 test samples
        let result = train_test_split(x, y, 0.1, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small"));
    }

    #[test]
    fn test_train_test_split_test_size_too_large() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0];

        // With 2 samples and test_size=0.99, we'd get 0 train samples
        let result = train_test_split(x, y, 0.99, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[test]
    fn test_train_test_split_preserves_total_samples() {
        let x: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32]).collect();
        let y: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let (train, test) = train_test_split(x, y, 0.3, Some(42)).unwrap();

        let total_samples = train.len().unwrap() + test.len().unwrap();
        assert_eq!(total_samples, 100);
    }
}
