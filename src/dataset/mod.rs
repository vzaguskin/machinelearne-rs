use std::{fmt::Debug, ops::Range};
use crate::backend::{Backend, Tensor1D, Tensor2D};
pub mod memory;
pub use self::memory::InMemoryDataset;

pub trait Dataset {
    type Error: Debug + 'static;
    type Item: ?Sized;

    /// Возвращает общее число сэмплов (если известно)
    fn len(&self) -> Option<usize>;

    /// Проверяет, пуст ли датасет
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Создаёт итератор по батчам заданного размера
    fn batches<'a, B: Backend>(&'a self, batch_size: usize) -> DatasetBatchIter<'a, B, Self>
    where
        Self: Sized,
    {
        DatasetBatchIter {
            dataset: self,
            batch_size,
            current: 0,
            _backend: std::marker::PhantomData,
        }
    }

    fn get_batch<B: Backend>(
        &self,
        range: Range<usize>,
    ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error>;
}

// Итератор по батчам
pub struct DatasetBatchIter<'a, B: Backend, D: ?Sized> {
    dataset: &'a D,
    batch_size: usize,
    current: usize,
    _backend: std::marker::PhantomData<B>,
}

impl<'a, B: Backend, D: Dataset> Iterator for DatasetBatchIter<'a, B, D> {
    type Item = Result<(Tensor2D<B>, Tensor1D<B>), D::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let total = self.dataset.len()?;
        if self.current >= total {
            return None;
        }

        let end = (self.current + self.batch_size).min(total);
        let range = self.current..end;
        self.current = end;

        // Запрашиваем подмножество данных и конвертируем в тензоры
        match self.dataset.get_batch::<B>(range) {
            Ok((x, y)) => Some(Ok((x, y))),
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use std::ops::Range;

    // Mock-датасет для тестирования логики итератора
    struct MockDataset {
        len: usize,
    }

    impl Dataset for MockDataset {
        type Error = &'static str;
        type Item = ();

        fn len(&self) -> Option<usize> {
            Some(self.len)
        }

        fn get_batch<B: Backend>(
            &self,
            range: Range<usize>,
        ) -> Result<(Tensor2D<B>, Tensor1D<B>), Self::Error> {
            if range.start >= self.len || range.end > self.len {
                return Err("range out of bounds");
            }

            let n = range.len();
            let start = range.start;

            // X: (n, 2) — делаем уникальные значения, например: [start*2, start*2+1, ...]
            let x_data: Vec<f32> = (0..n * 2)
                .map(|i| (start * 2 + i) as f32)
                .collect();
            let x = Tensor2D::<B>::new(x_data, n, 2);

            // y: (n,) — смещаем на start
            let y_data: Vec<f32> = (start..range.end).map(|i| i as f32).collect();
            let y = Tensor1D::<B>::new(y_data);

            Ok((x, y))
        }
    }

    #[test]
    fn test_dataset_is_empty() {
        let empty = MockDataset { len: 0 };
        assert!(empty.is_empty());

        let non_empty = MockDataset { len: 1 };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_batches_full() {
        let dataset = MockDataset { len: 6 };
        let mut iter = dataset.batches::<CpuBackend>(2);

        // Должно быть 3 полных батча
        for i in 0..3 {
            let batch = iter.next().unwrap().unwrap();
            let (x, y) = batch;
            assert_eq!(x.shape(), (2, 2));
            assert_eq!(y.to_vec(), vec![i as f64 * 2.0, i as f64 * 2.0 + 1.0]);
        }
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batches_partial_last() {
        let dataset = MockDataset { len: 5 };
        let mut iter = dataset.batches::<CpuBackend>(2);

        // 2 полных батча + 1 неполный
        assert_eq!(iter.next().unwrap().unwrap().0.shape(), (2, 2));
        assert_eq!(iter.next().unwrap().unwrap().0.shape(), (2, 2));
        assert_eq!(iter.next().unwrap().unwrap().0.shape(), (1, 2));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batches_larger_than_dataset() {
        let dataset = MockDataset { len: 3 };
        let mut iter = dataset.batches::<CpuBackend>(10);

        // Один батч на весь датасет
        let batch = iter.next().unwrap().unwrap();
        assert_eq!(batch.0.shape(), (3, 2));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batches_empty_dataset() {
        let dataset = MockDataset { len: 0 };
        let mut iter = dataset.batches::<CpuBackend>(2);
        assert!(iter.next().is_none());
    }
}