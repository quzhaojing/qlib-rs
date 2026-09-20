#[path = "../src/model_sequence.rs"]
mod measured_model_sequence;

use std::sync::{Arc, Mutex};

use measured_model_sequence::{
    ConcatDataset, EmptyConcatDatasetError, IndexSampler, ModelSequenceIndex, ModelSequenceLength,
};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Error {
    Empty(EmptyConcatDatasetError),
    Length(&'static str),
    Get(&'static str),
}

impl From<EmptyConcatDatasetError> for Error {
    fn from(error: EmptyConcatDatasetError) -> Self {
        Self::Empty(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Index {
    Integer(i128),
    Slice(Option<i128>, Option<i128>, Option<i128>),
}

type Item = Arc<Mutex<String>>;

struct Dataset {
    name: &'static str,
    length: usize,
    item: Item,
    length_error: bool,
    get_error: bool,
    events: Arc<Mutex<Vec<String>>>,
}

impl Dataset {
    fn new(name: &'static str, length: usize, events: &Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            name,
            length,
            item: Arc::new(Mutex::new(name.to_owned())),
            length_error: false,
            get_error: false,
            events: Arc::clone(events),
        }
    }

    fn failing_length(mut self) -> Self {
        self.length_error = true;
        self
    }

    fn failing_get(mut self) -> Self {
        self.get_error = true;
        self
    }
}

impl ModelSequenceLength<Error> for Dataset {
    fn len(&self) -> Result<usize, Error> {
        self.events
            .lock()
            .unwrap()
            .push(format!("len:{}", self.name));
        if self.length_error {
            Err(Error::Length(self.name))
        } else {
            Ok(self.length)
        }
    }
}

impl ModelSequenceIndex<Index, Item, Error> for Dataset {
    fn get(&self, index: &Index) -> Result<Item, Error> {
        self.events
            .lock()
            .unwrap()
            .push(format!("get:{}:{index:?}", self.name));
        if self.get_error {
            Err(Error::Get(self.name))
        } else {
            Ok(Arc::clone(&self.item))
        }
    }
}

#[test]
fn path_included_generic_instantiations_cover_complete_contract() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut concat = ConcatDataset::new(vec![
        Box::new(Dataset::new("three", 3, &events)),
        Box::new(Dataset::new("two", 2, &events)),
        Box::new(Dataset::new("four", 4, &events)),
    ]);
    assert_eq!(concat.dataset_count(), 3);
    assert_eq!(concat.datasets().len(), 3);
    assert_eq!(concat.len().unwrap(), 2);
    assert!(!concat.is_empty().unwrap());

    let slice = Index::Slice(None, Some(9), Some(-1));
    let items = concat.get(&slice).unwrap();
    assert_eq!(items.as_slice().len(), 3);
    let first = Arc::clone(&items.as_slice()[0]);
    *first.lock().unwrap() = "mutated".to_owned();
    assert_eq!(items.as_slice()[0].lock().unwrap().as_str(), "mutated");
    assert_eq!(items.into_items().len(), 3);

    let previous = concat.replace_datasets(vec![Box::new(Dataset::new("zero", 0, &events))]);
    assert_eq!(previous.len(), 3);
    assert!(concat.is_empty().unwrap());
    concat
        .datasets_mut()
        .push(Box::new(Dataset::new("one", 1, &events)));
    assert_eq!(concat.len().unwrap(), 0);
    concat.datasets_mut().clear();
    assert_eq!(concat.dataset_count(), 0);
    assert!(concat
        .get(&Index::Integer(10))
        .unwrap()
        .as_slice()
        .is_empty());
    assert_eq!(
        concat.len().unwrap_err(),
        Error::Empty(EmptyConcatDatasetError)
    );
    assert_eq!(
        concat.is_empty().unwrap_err(),
        Error::Empty(EmptyConcatDatasetError)
    );
    assert_eq!(
        EmptyConcatDatasetError.to_string(),
        "min() iterable argument is empty"
    );

    let length_failure = ConcatDataset::new(vec![
        Box::new(Dataset::new("ok", 1, &events)),
        Box::new(Dataset::new("bad-len", 1, &events).failing_length()),
        Box::new(Dataset::new("unreached", 1, &events)),
    ]);
    assert_eq!(length_failure.len().unwrap_err(), Error::Length("bad-len"));
    let get_failure = ConcatDataset::new(vec![
        Box::new(Dataset::new("ok", 1, &events)),
        Box::new(Dataset::new("bad-get", 1, &events).failing_get()),
        Box::new(Dataset::new("unreached", 1, &events)),
    ]);
    assert_eq!(
        get_failure.get(&Index::Integer(-1)).unwrap_err(),
        Error::Get("bad-get")
    );

    let mut sampler = IndexSampler::new(Dataset::new("sampler", 1, &events));
    assert_eq!(sampler.sampler().name, "sampler");
    assert_eq!(sampler.sampler_mut().length, 1);
    assert_eq!(sampler.len().unwrap(), 1);
    assert!(!sampler.is_empty().unwrap());
    let index = Index::Integer(i128::MAX);
    let (item, returned_index) = sampler.get(&index).unwrap();
    assert_eq!(item.lock().unwrap().as_str(), "sampler");
    assert!(std::ptr::eq(returned_index, &index));

    let previous = sampler.replace_sampler(Dataset::new("empty", 0, &events));
    assert_eq!(previous.name, "sampler");
    assert!(sampler.is_empty().unwrap());
    let previous = sampler.replace_sampler(Dataset::new("failure", 1, &events).failing_get());
    assert_eq!(previous.name, "empty");
    assert_eq!(
        sampler.get(&Index::Integer(0)).unwrap_err(),
        Error::Get("failure")
    );
}
