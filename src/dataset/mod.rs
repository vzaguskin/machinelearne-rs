use std::{fmt::Debug, ops::Range};
use crate::backend::{Backend, Tensor1D, Tensor2D};
pub mod memory;
pub use self::memory::InMemoryDataset;

pub trait Dataset {
    type Error: Debug;
    type Item: ?Sized;

    /// Возвращает общее число сэмплов (если известно)
    fn len(&self) -> Option<usize>;

    /// Проверяет, пуст ли датасет
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Создаёт итератор по батчам заданного размера
    fn batches<B: Backend>(&self, batch_size: usize) -> DatasetBatchIter<B, Self>
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