//! Ordered model-dataset composition corresponding to `qlib.model.utils`.

use std::fmt::{Display, Formatter};

/// Exact failure produced by Python's `min` when `ConcatDataset` has no children.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EmptyConcatDatasetError;

impl Display for EmptyConcatDatasetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("min() iterable argument is empty")
    }
}

impl std::error::Error for EmptyConcatDatasetError {}

/// Process-local boundary for a fallible Python-style dataset length.
pub trait ModelSequenceLength<Error> {
    /// # Errors
    /// Propagates the source dataset's length failure unchanged.
    fn len(&self) -> Result<usize, Error>;

    /// # Errors
    /// Propagates the source dataset's length failure unchanged.
    fn is_empty(&self) -> Result<bool, Error> {
        self.len().map(|length| length == 0)
    }
}

/// Process-local boundary for Python-style indexing without normalizing the index object.
///
/// Implementations decide how returned items retain identity. For mutable Python-like values,
/// an identity handle such as `Arc<Mutex<T>>` preserves later mutation visibility. The index is
/// borrowed so adapters can accept slices, arbitrary-precision integers, or identity handles.
pub trait ModelSequenceIndex<Index, Item, Error> {
    /// # Errors
    /// Propagates the source dataset's indexing failure unchanged.
    fn get(&self, index: &Index) -> Result<Item, Error>;
}

/// Object-safe combination used by `ConcatDataset` children.
pub trait ModelSequence<Index, Item, Error>:
    ModelSequenceLength<Error> + ModelSequenceIndex<Index, Item, Error>
{
}

impl<Sequence, Index, Item, Error> ModelSequence<Index, Item, Error> for Sequence where
    Sequence: ModelSequenceLength<Error> + ModelSequenceIndex<Index, Item, Error>
{
}

/// Fixed ordered result corresponding to the tuple returned by Python `ConcatDataset`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetTuple<Item> {
    items: Vec<Item>,
}

impl<Item> DatasetTuple<Item> {
    #[must_use]
    pub fn as_slice(&self) -> &[Item] {
        &self.items
    }

    #[must_use]
    pub fn into_items(self) -> Vec<Item> {
        self.items
    }
}

/// Variadic ordered dataset composition.
pub struct ConcatDataset<Index, Item, Error> {
    datasets: Vec<Box<dyn ModelSequence<Index, Item, Error>>>,
}

impl<Index, Item, Error> ConcatDataset<Index, Item, Error> {
    #[must_use]
    pub fn new(datasets: Vec<Box<dyn ModelSequence<Index, Item, Error>>>) -> Self {
        Self { datasets }
    }

    #[must_use]
    pub fn dataset_count(&self) -> usize {
        self.datasets.len()
    }

    #[must_use]
    pub fn datasets(&self) -> &[Box<dyn ModelSequence<Index, Item, Error>>] {
        &self.datasets
    }

    pub fn datasets_mut(&mut self) -> &mut Vec<Box<dyn ModelSequence<Index, Item, Error>>> {
        &mut self.datasets
    }

    pub fn replace_datasets(
        &mut self,
        datasets: Vec<Box<dyn ModelSequence<Index, Item, Error>>>,
    ) -> Vec<Box<dyn ModelSequence<Index, Item, Error>>> {
        std::mem::replace(&mut self.datasets, datasets)
    }
}

impl<Index, Item, Error> ModelSequenceLength<Error> for ConcatDataset<Index, Item, Error>
where
    Error: From<EmptyConcatDatasetError>,
{
    fn len(&self) -> Result<usize, Error> {
        let mut minimum = None;
        for dataset in &self.datasets {
            let length = dataset.len()?;
            minimum = Some(minimum.map_or(length, |current: usize| current.min(length)));
        }
        minimum.ok_or_else(|| EmptyConcatDatasetError.into())
    }
}

impl<Index, Item, Error> ModelSequenceIndex<Index, DatasetTuple<Item>, Error>
    for ConcatDataset<Index, Item, Error>
{
    fn get(&self, index: &Index) -> Result<DatasetTuple<Item>, Error> {
        let items = self
            .datasets
            .iter()
            .map(|dataset| dataset.get(index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DatasetTuple { items })
    }
}

/// Attaches the requested signed index to a dataset item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexSampler<Sampler> {
    sampler: Sampler,
}

impl<Sampler> IndexSampler<Sampler> {
    #[must_use]
    pub const fn new(sampler: Sampler) -> Self {
        Self { sampler }
    }

    #[must_use]
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    pub fn sampler_mut(&mut self) -> &mut Sampler {
        &mut self.sampler
    }

    pub fn replace_sampler(&mut self, sampler: Sampler) -> Sampler {
        std::mem::replace(&mut self.sampler, sampler)
    }

    /// # Errors
    /// Propagates the wrapped sampler's length failure unchanged.
    pub fn len<Error>(&self) -> Result<usize, Error>
    where
        Sampler: ModelSequenceLength<Error>,
    {
        self.sampler.len()
    }

    /// # Errors
    /// Propagates the wrapped sampler's length failure unchanged.
    pub fn is_empty<Error>(&self) -> Result<bool, Error>
    where
        Sampler: ModelSequenceLength<Error>,
    {
        self.sampler.is_empty()
    }

    /// Returns the exact borrowed index object alongside the delegated item.
    ///
    /// # Errors
    /// Propagates the wrapped sampler's indexing failure unchanged.
    pub fn get<'index, Index, Item, Error>(
        &self,
        index: &'index Index,
    ) -> Result<(Item, &'index Index), Error>
    where
        Sampler: ModelSequenceIndex<Index, Item, Error>,
    {
        self.sampler.get(index).map(|item| (item, index))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{
        ConcatDataset, EmptyConcatDatasetError, IndexSampler, ModelSequenceIndex,
        ModelSequenceLength,
    };

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Error {
        Empty(EmptyConcatDatasetError),
        Source(&'static str),
    }

    impl From<EmptyConcatDatasetError> for Error {
        fn from(error: EmptyConcatDatasetError) -> Self {
            Self::Empty(error)
        }
    }

    type Item = Arc<Mutex<String>>;

    struct Dataset {
        length: Result<usize, Error>,
        item: Result<Item, Error>,
    }

    impl Dataset {
        fn value(name: &str, length: usize) -> Self {
            Self {
                length: Ok(length),
                item: Ok(Arc::new(Mutex::new(name.to_owned()))),
            }
        }

        fn failure(message: &'static str) -> Self {
            Self {
                length: Err(Error::Source(message)),
                item: Err(Error::Source(message)),
            }
        }
    }

    impl ModelSequenceLength<Error> for Dataset {
        fn len(&self) -> Result<usize, Error> {
            self.length.clone()
        }
    }

    impl ModelSequenceIndex<i128, Item, Error> for Dataset {
        fn get(&self, _index: &i128) -> Result<Item, Error> {
            self.item.clone()
        }
    }

    #[test]
    fn unit_instantiations_cover_generic_success_empty_replacement_and_failure_paths() {
        let mut concat = ConcatDataset::new(vec![
            Box::new(Dataset::value("three", 3)),
            Box::new(Dataset::value("two", 2)),
            Box::new(Dataset::value("four", 4)),
        ]);
        assert_eq!(concat.dataset_count(), 3);
        assert_eq!(concat.datasets().len(), 3);
        assert_eq!(concat.len(), Ok(2));
        assert_eq!(concat.is_empty(), Ok(false));
        let index = i128::MAX;
        let items = concat.get(&index).unwrap();
        assert_eq!(items.as_slice().len(), 3);
        assert_eq!(items.into_items().len(), 3);

        let previous = concat.replace_datasets(vec![Box::new(Dataset::value("zero", 0))]);
        assert_eq!(previous.len(), 3);
        assert_eq!(concat.is_empty(), Ok(true));
        concat
            .datasets_mut()
            .push(Box::new(Dataset::value("one", 1)));
        assert_eq!(concat.len(), Ok(0));
        concat.datasets_mut().clear();
        assert!(concat.get(&index).unwrap().as_slice().is_empty());
        assert_eq!(concat.len(), Err(Error::Empty(EmptyConcatDatasetError)));
        assert_eq!(
            concat.is_empty(),
            Err(Error::Empty(EmptyConcatDatasetError))
        );
        assert_eq!(
            EmptyConcatDatasetError.to_string(),
            "min() iterable argument is empty"
        );

        let failed = ConcatDataset::new(vec![
            Box::new(Dataset::value("ok", 1)),
            Box::new(Dataset::failure("source")),
        ]);
        assert_eq!(failed.len(), Err(Error::Source("source")));
        assert_eq!(failed.get(&index).unwrap_err(), Error::Source("source"));

        let mut sampler = IndexSampler::new(Dataset::value("sampler", 1));
        assert_eq!(sampler.sampler().length, Ok(1));
        assert_eq!(sampler.sampler_mut().length, Ok(1));
        assert_eq!(sampler.len(), Ok(1));
        assert_eq!(sampler.is_empty(), Ok(false));
        let (item, returned_index) = sampler.get(&index).unwrap();
        assert_eq!(item.lock().unwrap().as_str(), "sampler");
        assert!(std::ptr::eq(returned_index, &index));
        let previous = sampler.replace_sampler(Dataset::value("empty", 0));
        assert_eq!(previous.length, Ok(1));
        assert_eq!(sampler.is_empty(), Ok(true));
        sampler.replace_sampler(Dataset::failure("sampler-source"));
        assert_eq!(sampler.len(), Err(Error::Source("sampler-source")));
        assert_eq!(
            sampler.get(&index).unwrap_err(),
            Error::Source("sampler-source")
        );
    }
}
