//! Cross-validation utilities.

use rand::prelude::*;

/// Trait for cross-validation splitters.
///
/// This trait is object-safe, allowing dynamic dispatch via `Box<dyn CVSplit>`.
pub trait CVSplit {
    /// Generate train/test index splits for n samples.
    ///
    /// Returns a vector of (train_indices, test_indices) tuples, one for each fold.
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;

    /// Number of splits (folds).
    fn n_splits(&self) -> usize;
}

/// K-Fold cross-validation.
///
/// Splits data into k consecutive folds. Each fold is used once as validation
/// while the k-1 remaining folds form the training set.
///
/// # Example
///
/// ```rust
/// use machinelearne_rs::model_selection::{CVSplit, KFold};
///
/// let kfold = KFold::new(5).with_random_state(42);
/// let splits = kfold.split(100);
///
/// assert_eq!(splits.len(), 5);
/// for (train, test) in splits {
///     assert_eq!(train.len() + test.len(), 100);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct KFold {
    /// Number of folds.
    pub n_splits: usize,
    /// Whether to shuffle before splitting.
    pub shuffle: bool,
    /// Random seed for shuffling.
    pub random_state: Option<u64>,
}

impl KFold {
    /// Create a new K-Fold splitter.
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds. Must be >= 2.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable shuffling with a random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.shuffle = true;
        self.random_state = Some(seed);
        self
    }
}

impl CVSplit for KFold {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert!(self.n_splits >= 2, "n_splits must be >= 2");
        assert!(
            n_samples >= self.n_splits,
            "n_samples ({}) must be >= n_splits ({})",
            n_samples,
            self.n_splits
        );

        // Create indices
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if self.shuffle {
            if let Some(seed) = self.random_state {
                let mut rng = SmallRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            }
        }

        // Compute fold sizes
        let fold_sizes: Vec<usize> = (0..self.n_splits)
            .map(|i| {
                let n_samples_per_fold = n_samples / self.n_splits;
                if i < (n_samples % self.n_splits) {
                    n_samples_per_fold + 1
                } else {
                    n_samples_per_fold
                }
            })
            .collect();

        let mut splits = Vec::with_capacity(self.n_splits);
        let mut current = 0;

        for &fold_size in &fold_sizes {
            let test_start = current;
            let test_end = current + fold_size;

            // Build test indices
            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();

            // Build train indices (all other indices)
            let mut train_indices = Vec::with_capacity(n_samples - fold_size);
            train_indices.extend_from_slice(&indices[..test_start]);
            train_indices.extend_from_slice(&indices[test_end..]);

            splits.push((train_indices, test_indices));
            current = test_end;
        }

        splits
    }

    fn n_splits(&self) -> usize {
        self.n_splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_basic() {
        let kfold = KFold::new(3);
        let splits = kfold.split(9);

        assert_eq!(splits.len(), 3);

        for (train, test) in &splits {
            assert_eq!(train.len(), 6);
            assert_eq!(test.len(), 3);
        }
    }

    #[test]
    fn test_kfold_uneven_folds() {
        // 10 samples into 3 folds: sizes should be [4, 3, 3]
        let kfold = KFold::new(3);
        let splits = kfold.split(10);

        assert_eq!(splits.len(), 3);

        // First fold gets one extra sample
        assert_eq!(splits[0].1.len(), 4); // test
        assert_eq!(splits[0].0.len(), 6); // train

        assert_eq!(splits[1].1.len(), 3); // test
        assert_eq!(splits[1].0.len(), 7); // train

        assert_eq!(splits[2].1.len(), 3); // test
        assert_eq!(splits[2].0.len(), 7); // train
    }

    #[test]
    fn test_kfold_all_samples_used() {
        let kfold = KFold::new(5);
        let n_samples = 20;
        let splits = kfold.split(n_samples);

        // Collect all test indices
        let mut all_test_indices: Vec<usize> = splits
            .iter()
            .flat_map(|(_, test)| test.iter().copied())
            .collect();
        all_test_indices.sort();
        all_test_indices.dedup();

        // Every sample should appear exactly once in some test set
        assert_eq!(all_test_indices.len(), n_samples);
        assert_eq!(all_test_indices, (0..n_samples).collect::<Vec<_>>());
    }

    #[test]
    fn test_kfold_no_overlap() {
        let kfold = KFold::new(5);
        let splits = kfold.split(20);

        // Within each split, train and test should be disjoint
        for (train, test) in &splits {
            for &t in test {
                assert!(!train.contains(&t), "Test index {} found in train set", t);
            }
        }
    }

    #[test]
    fn test_kfold_shuffle_reproducible() {
        let kfold1 = KFold::new(3).with_random_state(42);
        let kfold2 = KFold::new(3).with_random_state(42);

        let splits1 = kfold1.split(10);
        let splits2 = kfold2.split(10);

        for ((train1, test1), (train2, test2)) in splits1.iter().zip(splits2.iter()) {
            assert_eq!(train1, train2);
            assert_eq!(test1, test2);
        }
    }

    #[test]
    fn test_kfold_shuffle_different_seeds() {
        let kfold1 = KFold::new(3).with_random_state(1);
        let kfold2 = KFold::new(3).with_random_state(2);

        let splits1 = kfold1.split(10);
        let splits2 = kfold2.split(10);

        // At least one fold should have different indices
        let any_different = splits1
            .iter()
            .zip(splits2.iter())
            .any(|((t1, _), (t2, _))| t1 != t2);
        assert!(any_different);
    }

    #[test]
    fn test_kfold_no_shuffle_ordered() {
        let kfold = KFold::new(3);
        let splits = kfold.split(9);

        // Without shuffle, test sets should be consecutive
        assert_eq!(splits[0].1, vec![0, 1, 2]);
        assert_eq!(splits[1].1, vec![3, 4, 5]);
        assert_eq!(splits[2].1, vec![6, 7, 8]);
    }

    #[test]
    #[should_panic(expected = "n_splits must be >= 2")]
    fn test_kfold_too_few_splits() {
        let kfold = KFold::new(1);
        kfold.split(10);
    }

    #[test]
    #[should_panic(expected = "n_samples")]
    fn test_kfold_too_few_samples() {
        let kfold = KFold::new(5);
        kfold.split(3);
    }

    #[test]
    fn test_kfold_n_splits() {
        let kfold = KFold::new(10);
        assert_eq!(kfold.n_splits(), 10);
    }

    #[test]
    fn test_kfold_each_sample_tested_once() {
        let kfold = KFold::new(5);
        let n_samples = 25;
        let splits = kfold.split(n_samples);

        // Count how many times each sample appears in test sets
        let mut test_counts = vec![0; n_samples];
        for (_, test) in &splits {
            for &idx in test {
                test_counts[idx] += 1;
            }
        }

        // Each sample should appear exactly once
        for (idx, &count) in test_counts.iter().enumerate() {
            assert_eq!(
                count, 1,
                "Sample {} appeared {} times in test sets",
                idx, count
            );
        }
    }
}
