use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use domain_core::model_sequence::{
    ConcatDataset, DatasetTuple, EmptyConcatDatasetError, IndexSampler, ModelSequenceIndex,
    ModelSequenceLength,
};
use serde_json::{json, Value};

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProbeError {
    Empty(EmptyConcatDatasetError),
    Get(String),
    Length(String),
    OutOfRange(i64),
}

impl From<EmptyConcatDatasetError> for ProbeError {
    fn from(error: EmptyConcatDatasetError) -> Self {
        Self::Empty(error)
    }
}

type Item = Arc<Mutex<String>>;

struct ProbeDataset {
    name: &'static str,
    values: Vec<Item>,
    events: Arc<Mutex<Vec<String>>>,
    length_error: Option<&'static str>,
    get_error: Option<i64>,
}

#[derive(Debug, Eq, PartialEq)]
enum RichIndex {
    Slice {
        start: Option<i128>,
        stop: Option<i128>,
        step: Option<i128>,
    },
    BigInteger(String),
}

struct EchoDataset;

impl ModelSequenceLength<ProbeError> for EchoDataset {
    fn len(&self) -> Result<usize, ProbeError> {
        Ok(7)
    }
}

impl ModelSequenceIndex<Arc<RichIndex>, Arc<RichIndex>, ProbeError> for EchoDataset {
    fn get(&self, index: &Arc<RichIndex>) -> Result<Arc<RichIndex>, ProbeError> {
        Ok(Arc::clone(index))
    }
}

impl ProbeDataset {
    fn new(name: &'static str, values: &[&str], events: &Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            name,
            values: values
                .iter()
                .map(|value| Arc::new(Mutex::new((*value).to_owned())))
                .collect(),
            events: Arc::clone(events),
            length_error: None,
            get_error: None,
        }
    }

    fn with_length_error(mut self, message: &'static str) -> Self {
        self.length_error = Some(message);
        self
    }

    fn with_get_error(mut self, index: i64) -> Self {
        self.get_error = Some(index);
        self
    }
}

impl ModelSequenceLength<ProbeError> for ProbeDataset {
    fn len(&self) -> Result<usize, ProbeError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("len:{}", self.name));
        self.length_error.map_or(Ok(self.values.len()), |message| {
            Err(ProbeError::Length(message.to_owned()))
        })
    }
}

impl ModelSequenceIndex<i64, Item, ProbeError> for ProbeDataset {
    fn get(&self, index: &i64) -> Result<Item, ProbeError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("get:{}:{index}", self.name));
        if self.get_error == Some(*index) {
            return Err(ProbeError::Get(format!(
                "get failure {} {index}",
                self.name
            )));
        }
        let resolved = if *index < 0 {
            i64::try_from(self.values.len()).unwrap() + *index
        } else {
            *index
        };
        usize::try_from(resolved)
            .ok()
            .and_then(|resolved| self.values.get(resolved))
            .cloned()
            .ok_or(ProbeError::OutOfRange(*index))
    }
}

fn fixture() -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_MODEL_UTILS").map_or_else(
        || root.join("../../../qlib/qlib/model/utils.py"),
        PathBuf::from,
    );
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/model_sequence_contract.py"))
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn item_values(items: &DatasetTuple<Item>) -> Vec<String> {
    items
        .as_slice()
        .iter()
        .map(|item| item.lock().unwrap().clone())
        .collect()
}

#[test]
fn live_source_surface_inheritance_and_failures_are_hash_pinned() {
    let source = fixture();
    assert_eq!(
        source["source_sha256"],
        "4adc4cea4e29287c98967e143bc9a143da561d33d3539e774dff44636486949f"
    );
    assert_eq!(
        source["surface"],
        json!(["ConcatDataset", "Dataset", "IndexSampler"])
    );
    assert_eq!(source["concat_base_is_dataset"], true);
    assert_eq!(source["index_sampler_base"], "object");
    assert_eq!(source["constructor_tuple"], "tuple");
    assert_eq!(source["constructor_identity"], json!([true, true]));
    assert_eq!(source["normal_tuple"], "tuple");
    assert_eq!(source["normal_identity"], json!([true, true]));
    assert_eq!(
        source["empty_len"],
        json!({"type": "ValueError", "message": "min() iterable argument is empty"})
    );
    assert_eq!(source["empty_get_type"], "tuple");
    assert_eq!(source["empty_get"], json!([]));
}

#[test]
fn concat_preserves_order_signed_indices_identity_and_mutation_visibility() {
    let source = fixture();
    let events = Arc::new(Mutex::new(Vec::new()));
    let first = ProbeDataset::new("a", &["a0", "a1", "a2"], &events);
    let first_identity = Arc::clone(&first.values[1]);
    let second = ProbeDataset::new("b", &["b0", "b1"], &events);
    let second_identity = Arc::clone(&second.values[1]);
    let concat = ConcatDataset::new(vec![Box::new(first), Box::new(second)]);
    assert_eq!(concat.dataset_count(), 2);

    let normal = concat.get(&1).unwrap();
    assert!(Arc::ptr_eq(&normal.as_slice()[0], &first_identity));
    assert!(Arc::ptr_eq(&normal.as_slice()[1], &second_identity));
    *first_identity.lock().unwrap() = "a1-mutated".to_owned();
    assert_eq!(
        item_values(&normal),
        source["normal_after_mutation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    );
    assert_eq!(*events.lock().unwrap(), ["get:a:1", "get:b:1"]);

    events.lock().unwrap().clear();
    let negative = concat.get(&-1).unwrap();
    assert_eq!(item_values(&negative), ["a2", "b1"]);
    assert_eq!(*events.lock().unwrap(), ["get:a:-1", "get:b:-1"]);
    assert_eq!(negative.into_items().len(), 2);
}

#[test]
fn concat_length_and_source_failures_retain_order_and_exact_error_values() {
    let source = fixture();
    let events = Arc::new(Mutex::new(Vec::new()));
    let concat = ConcatDataset::new(vec![
        Box::new(ProbeDataset::new("a", &["a0", "a1", "a2"], &events)),
        Box::new(ProbeDataset::new("b", &["b0", "b1"], &events)),
    ]);
    assert_eq!(
        concat.len().unwrap(),
        source["length"].as_u64().unwrap() as usize
    );
    assert_eq!(*events.lock().unwrap(), ["len:a", "len:b"]);
    events.lock().unwrap().clear();
    assert!(!concat.is_empty().unwrap());
    assert_eq!(*events.lock().unwrap(), ["len:a", "len:b"]);

    events.lock().unwrap().clear();
    let failed_get = ConcatDataset::new(vec![
        Box::new(ProbeDataset::new("a", &["a0", "a1"], &events)),
        Box::new(ProbeDataset::new("bad", &[], &events).with_get_error(1)),
    ]);
    assert_eq!(
        failed_get.get(&1).unwrap_err(),
        ProbeError::Get("get failure bad 1".to_owned())
    );
    assert_eq!(*events.lock().unwrap(), ["get:a:1", "get:bad:1"]);
    assert_eq!(source["get_failure"]["type"], "ProbeError");

    events.lock().unwrap().clear();
    let failed_length = ConcatDataset::new(vec![
        Box::new(ProbeDataset::new("a", &["a0"], &events)),
        Box::new(ProbeDataset::new("bad", &[], &events).with_length_error("len failure bad")),
    ]);
    assert_eq!(
        failed_length.len().unwrap_err(),
        ProbeError::Length("len failure bad".to_owned())
    );
    assert_eq!(*events.lock().unwrap(), ["len:a", "len:bad"]);

    events.lock().unwrap().clear();
    assert_eq!(concat.get(&9).unwrap_err(), ProbeError::OutOfRange(9));
    assert_eq!(*events.lock().unwrap(), ["get:a:9"]);
    assert_eq!(source["out_of_range"]["type"], "IndexError");
}

#[test]
fn empty_concat_and_index_sampler_match_source_boundaries() {
    let source = fixture();
    let empty = ConcatDataset::<i64, Item, ProbeError>::new(Vec::new());
    assert_eq!(empty.dataset_count(), 0);
    assert!(empty.get(&123).unwrap().as_slice().is_empty());
    let error = empty.len().unwrap_err();
    assert_eq!(error, ProbeError::Empty(EmptyConcatDatasetError));
    assert_eq!(error, EmptyConcatDatasetError.into());
    assert_eq!(
        empty.is_empty().unwrap_err(),
        ProbeError::Empty(EmptyConcatDatasetError)
    );
    assert_eq!(
        EmptyConcatDatasetError.to_string(),
        source["empty_len"]["message"]
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let dataset = ProbeDataset::new("a", &["a0", "a1", "a2"], &events);
    let last_identity = Arc::clone(&dataset.values[2]);
    let sampler = IndexSampler::new(dataset);
    assert_eq!(sampler.sampler().name, "a");
    let requested = -1;
    let (item, index) = sampler.get(&requested).unwrap();
    assert!(Arc::ptr_eq(&item, &last_identity));
    assert!(std::ptr::eq(index, &requested));
    assert_eq!(*events.lock().unwrap(), ["get:a:-1"]);

    events.lock().unwrap().clear();
    assert_eq!(sampler.get(&9).unwrap_err(), ProbeError::OutOfRange(9));
    assert_eq!(*events.lock().unwrap(), ["get:a:9"]);
    events.lock().unwrap().clear();
    assert_eq!(sampler.len().unwrap(), 3);
    assert_eq!(*events.lock().unwrap(), ["len:a"]);
    events.lock().unwrap().clear();
    assert!(!sampler.is_empty().unwrap());
    assert_eq!(*events.lock().unwrap(), ["len:a"]);
    assert_eq!(source["sampler_index"], -1);
    assert_eq!(source["sampler_identity"], true);

    events.lock().unwrap().clear();
    let empty_sampler = IndexSampler::new(ProbeDataset::new("empty", &[], &events));
    assert!(empty_sampler.is_empty().unwrap());
}

#[test]
fn arbitrary_index_identity_and_mutable_public_attributes_have_native_boundaries() {
    let source = fixture();
    assert_eq!(source["opaque_index_identity"], json!([true, true]));
    assert_eq!(source["slice_index_identity"], json!([true, true]));
    assert_eq!(source["sampler_index_identity"], json!([true, true]));

    for index in [
        Arc::new(RichIndex::Slice {
            start: None,
            stop: None,
            step: Some(-1),
        }),
        Arc::new(RichIndex::BigInteger(
            "10000000000000000000000000000000000000000".to_owned(),
        )),
    ] {
        let concat = ConcatDataset::new(vec![Box::new(EchoDataset), Box::new(EchoDataset)]);
        let result = concat.get(&index).unwrap();
        assert!(result
            .as_slice()
            .iter()
            .all(|item| Arc::ptr_eq(item, &index)));
        let sampler = IndexSampler::new(EchoDataset);
        let (item, returned_index) = sampler.get(&index).unwrap();
        assert!(Arc::ptr_eq(&item, &index));
        assert!(std::ptr::eq(returned_index, &index));
    }

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut concat = ConcatDataset::new(vec![Box::new(ProbeDataset::new(
        "a",
        &["a0", "a1", "a2"],
        &events,
    ))]);
    assert_eq!(concat.datasets().len(), 1);
    let previous = concat.replace_datasets(vec![
        Box::new(ProbeDataset::new("b", &["b0", "b1"], &events)),
        Box::new(ProbeDataset::new("a", &["a0", "a1", "a2"], &events)),
    ]);
    assert_eq!(previous.len(), 1);
    assert_eq!(concat.dataset_count(), 2);
    assert_eq!(concat.len().unwrap(), 2);
    concat
        .datasets_mut()
        .push(Box::new(ProbeDataset::new("empty", &[], &events)));
    assert!(concat.is_empty().unwrap());
    concat.datasets_mut().clear();
    assert_eq!(concat.len().unwrap_err(), EmptyConcatDatasetError.into());

    let mut sampler = IndexSampler::new(ProbeDataset::new("a", &["a0"], &events));
    assert_eq!(sampler.sampler_mut().name, "a");
    let previous = sampler.replace_sampler(ProbeDataset::new("b", &["b0", "b1"], &events));
    assert_eq!(previous.name, "a");
    assert_eq!(sampler.sampler().name, "b");
    assert_eq!(sampler.len().unwrap(), 2);
    assert_eq!(source["replacement_values"], json!(["b0", "a0"]));
    assert_eq!(source["replacement_length"], 2);
    assert_eq!(source["replacement_sampler_value"], "b0");
    assert_eq!(source["replacement_sampler_index"], 0);
}
