//! Ephemeral producer/consumer queue used by RL trainer and backtest orchestration.

use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crossbeam_channel::{Receiver, SendTimeoutError, Sender, bounded, select};
use rand::seq::SliceRandom;
use thiserror::Error;

const SEND_POLL_INTERVAL: Duration = Duration::from_millis(10);
const FIRST_GET_TIMEOUT: Duration = Duration::from_secs(5);
const GET_TIMEOUT: Duration = Duration::from_millis(500);

/// Process-local dataset seam corresponding to Python's `Sequence.__len__/__getitem__` contract.
pub trait DataQueueDataset<T>: Send + Sync {
    fn len(&self) -> usize;

    /// # Errors
    /// Returns a dataset-specific loading failure for `index`.
    fn get(&self, index: usize) -> Result<T, String>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> DataQueueDataset<T> for Vec<T>
where
    T: Clone + Send + Sync,
{
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn get(&self, index: usize) -> Result<T, String> {
        self.as_slice()
            .get(index)
            .cloned()
            .ok_or_else(|| format!("dataset index {index} is out of bounds"))
    }
}

/// Replaceable epoch sampler. Returned indices are emitted in exactly this order.
pub trait DataQueueSampler: Send {
    /// # Errors
    /// Returns a sampler-specific failure for this epoch.
    fn indices(&mut self, dataset_len: usize, epoch: u64) -> Result<Vec<usize>, String>;
}

#[derive(Debug, Default)]
pub struct SequentialDataQueueSampler;

impl DataQueueSampler for SequentialDataQueueSampler {
    fn indices(&mut self, dataset_len: usize, _epoch: u64) -> Result<Vec<usize>, String> {
        Ok((0..dataset_len).collect())
    }
}

#[derive(Debug, Default)]
pub struct RandomDataQueueSampler;

impl DataQueueSampler for RandomDataQueueSampler {
    fn indices(&mut self, dataset_len: usize, _epoch: u64) -> Result<Vec<usize>, String> {
        let mut indices: Vec<_> = (0..dataset_len).collect();
        indices.shuffle(&mut rand::rng());
        Ok(indices)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataQueueConfig {
    pub repeat: i64,
    pub shuffle: bool,
    pub producer_num_workers: usize,
    pub queue_maxsize: usize,
}

impl Default for DataQueueConfig {
    fn default() -> Self {
        Self {
            repeat: 1,
            shuffle: true,
            producer_num_workers: 0,
            queue_maxsize: 0,
        }
    }
}

impl DataQueueConfig {
    #[must_use]
    pub fn effective_queue_maxsize(self) -> usize {
        if self.queue_maxsize == 0 {
            thread::available_parallelism().map_or(1, std::num::NonZero::get)
        } else {
            self.queue_maxsize
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DataQueueProducerError {
    #[error("data queue sampler failed at epoch {epoch}: {message}")]
    Sampler { epoch: u64, message: String },
    #[error("data queue sampler returned out-of-range index {index} for dataset length {len}")]
    InvalidSampleIndex { index: usize, len: usize },
    #[error("data queue dataset failed at index {index}: {message}")]
    Dataset { index: usize, message: String },
    #[error("data queue consumer disconnected")]
    ConsumerDisconnected,
    #[error("data queue producer panicked")]
    Panicked,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DataQueueError {
    #[error("data queue can not activate twice")]
    AlreadyActivated,
    #[error("data queue must be activated before iteration")]
    NotActivated,
    #[error("data queue has already been cleaned up")]
    Closed,
    #[error("data queue is exhausted")]
    Exhausted,
    #[error(transparent)]
    Producer(#[from] DataQueueProducerError),
}

/// An activated queue is intentionally single-use, matching Qlib's ephemeral `DataQueue`.
pub struct DataQueue<T> {
    receiver: Receiver<T>,
    completion: Receiver<()>,
    completion_sender: Option<Sender<()>>,
    sender: Option<Sender<T>>,
    dataset: Arc<dyn DataQueueDataset<T>>,
    sampler: Option<Box<dyn DataQueueSampler>>,
    config: DataQueueConfig,
    cancelled: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    producer_error: Arc<Mutex<Option<DataQueueProducerError>>>,
    producer: Option<JoinHandle<()>>,
    activated: bool,
    first_get: bool,
}

impl<T> DataQueue<T>
where
    T: Send + 'static,
{
    #[must_use]
    pub fn new(dataset: Arc<dyn DataQueueDataset<T>>, config: DataQueueConfig) -> Self {
        let sampler: Box<dyn DataQueueSampler> = if config.shuffle {
            Box::new(RandomDataQueueSampler)
        } else {
            Box::new(SequentialDataQueueSampler)
        };
        Self::with_sampler(dataset, config, sampler)
    }

    #[must_use]
    pub fn with_sampler(
        dataset: Arc<dyn DataQueueDataset<T>>,
        config: DataQueueConfig,
        sampler: Box<dyn DataQueueSampler>,
    ) -> Self {
        let (sender, receiver) = bounded(config.effective_queue_maxsize());
        let (completion_sender, completion) = bounded(1);
        Self {
            receiver,
            completion,
            completion_sender: Some(completion_sender),
            sender: Some(sender),
            dataset,
            sampler: Some(sampler),
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
            done: Arc::new(AtomicBool::new(false)),
            producer_error: Arc::new(Mutex::new(None)),
            producer: None,
            activated: false,
            first_get: true,
        }
    }

    /// Starts the single producer thread.
    ///
    /// # Errors
    /// Returns [`DataQueueError::AlreadyActivated`] after the first activation attempt.
    pub fn activate(&mut self) -> Result<&mut Self, DataQueueError> {
        if self.activated {
            return Err(DataQueueError::AlreadyActivated);
        }
        self.activated = true;
        let sender = self.sender.as_ref().ok_or(DataQueueError::Closed)?.clone();
        let completion_sender = self
            .completion_sender
            .take()
            .ok_or(DataQueueError::Closed)?;
        let dataset = Arc::clone(&self.dataset);
        let mut sampler = self.sampler.take().ok_or(DataQueueError::Closed)?;
        let config = self.config;
        let cancelled = Arc::clone(&self.cancelled);
        let done = Arc::clone(&self.done);
        let producer_error = Arc::clone(&self.producer_error);
        self.producer = Some(thread::spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| {
                produce(&dataset, sampler.as_mut(), config, &sender, &cancelled)
            }));
            let failure = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(_) => Some(DataQueueProducerError::Panicked),
            };
            if let Some(error) = failure {
                *lock_unpoisoned(&producer_error) = Some(error);
            }
            done.store(true, Ordering::Release);
            let _ = completion_sender.send(());
        }));
        Ok(self)
    }

    #[must_use]
    pub fn is_activated(&self) -> bool {
        self.activated
    }

    #[must_use]
    pub fn done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn producer_error(&self) -> Option<DataQueueProducerError> {
        lock_unpoisoned(&self.producer_error).clone()
    }

    /// Adds an item through the same bounded queue, matching Python's public `put` helper.
    ///
    /// # Errors
    /// Returns when cleanup or receiver disconnection makes delivery impossible.
    pub fn put(&self, value: T) -> Result<(), DataQueueError> {
        send_cancellable(
            self.sender
                .as_ref()
                .ok_or(DataQueueProducerError::ConsumerDisconnected)?,
            value,
            &self.cancelled,
        )
        .map_err(Into::into)
    }

    /// Waits until an item arrives or the producer is definitively exhausted.
    ///
    /// # Errors
    /// Returns the producer failure, or [`DataQueueError::Exhausted`] after normal completion.
    pub fn get(&mut self) -> Result<T, DataQueueError> {
        let timeout = if self.first_get {
            self.first_get = false;
            FIRST_GET_TIMEOUT
        } else {
            GET_TIMEOUT
        };
        loop {
            if self.done() {
                return self.receiver.try_recv().map_err(|_| self.terminal_error());
            }
            if let Ok(value) = self.receiver.try_recv() {
                return Ok(value);
            }
            select! {
                recv(self.receiver) -> value => {
                    match value {
                        Ok(value) => return Ok(value),
                        Err(_) => return Err(self.terminal_error()),
                    }
                }
                recv(self.completion) -> _ => {}
                default(timeout) => {}
            }
        }
    }

    /// Creates the consuming iterator after activation.
    ///
    /// # Errors
    /// Returns [`DataQueueError::NotActivated`] before [`Self::activate`].
    pub fn consumer(&mut self) -> Result<DataQueueIter<'_, T>, DataQueueError> {
        if !self.activated {
            return Err(DataQueueError::NotActivated);
        }
        Ok(DataQueueIter {
            queue: self,
            finished: false,
        })
    }

    /// Cancels production, drains queued values, and joins the producer.
    pub fn cleanup(&mut self) {
        self.cleanup_with_join(|producer| {
            let _ = producer.join();
        });
    }

    // Keep the join boundary explicit so an in-flight producer can be released after the
    // initial drain in lifecycle tests. Production uses the ordinary blocking join above.
    fn cleanup_with_join(&mut self, join: impl FnOnce(JoinHandle<()>)) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        drain(&self.receiver);
        if let Some(producer) = self.producer.take() {
            join(producer);
        }
        drain(&self.receiver);
        self.done.store(true, Ordering::Release);
    }

    fn terminal_error(&self) -> DataQueueError {
        self.producer_error()
            .map_or(DataQueueError::Exhausted, DataQueueError::Producer)
    }
}

impl<T> Drop for DataQueue<T> {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        drain(&self.receiver);
        if let Some(producer) = self.producer.take() {
            let _ = producer.join();
        }
    }
}

pub struct DataQueueIter<'a, T>
where
    T: Send + 'static,
{
    queue: &'a mut DataQueue<T>,
    finished: bool,
}

impl<T> Iterator for DataQueueIter<'_, T>
where
    T: Send + 'static,
{
    type Item = Result<T, DataQueueError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.queue.get() {
            Ok(value) => Some(Ok(value)),
            Err(DataQueueError::Exhausted) => {
                self.finished = true;
                None
            }
            Err(error) => {
                self.finished = true;
                Some(Err(error))
            }
        }
    }
}

fn produce<T>(
    dataset: &Arc<dyn DataQueueDataset<T>>,
    sampler: &mut dyn DataQueueSampler,
    config: DataQueueConfig,
    output: &Sender<T>,
    cancelled: &AtomicBool,
) -> Result<(), DataQueueProducerError>
where
    T: Send + 'static,
{
    let repeats = if config.repeat == -1 {
        None
    } else {
        Some(u64::try_from(config.repeat.max(0)).unwrap_or(0))
    };
    let mut epoch = 0_u64;
    while repeats.is_none_or(|limit| epoch < limit) && !cancelled.load(Ordering::Acquire) {
        let indices = sampler
            .indices(dataset.len(), epoch)
            .map_err(|message| DataQueueProducerError::Sampler { epoch, message })?;
        if let Some(index) = indices
            .iter()
            .copied()
            .find(|index| *index >= dataset.len())
        {
            return Err(DataQueueProducerError::InvalidSampleIndex {
                index,
                len: dataset.len(),
            });
        }
        if config.producer_num_workers == 0 {
            produce_epoch_sequential(dataset.as_ref(), &indices, output, cancelled)?;
        } else {
            produce_epoch_parallel(
                dataset,
                &indices,
                config.producer_num_workers,
                config.effective_queue_maxsize(),
                output,
                cancelled,
            )?;
        }
        epoch = epoch.saturating_add(1);
    }
    Ok(())
}

fn produce_epoch_sequential<T>(
    dataset: &dyn DataQueueDataset<T>,
    indices: &[usize],
    output: &Sender<T>,
    cancelled: &AtomicBool,
) -> Result<(), DataQueueProducerError>
where
    T: Send + 'static,
{
    for &index in indices {
        if cancelled.load(Ordering::Acquire) {
            break;
        }
        let value = dataset
            .get(index)
            .map_err(|message| DataQueueProducerError::Dataset { index, message })?;
        send_cancellable(output, value, cancelled)?;
    }
    Ok(())
}

fn produce_epoch_parallel<T>(
    dataset: &Arc<dyn DataQueueDataset<T>>,
    indices: &[usize],
    worker_count: usize,
    queue_capacity: usize,
    output: &Sender<T>,
    cancelled: &AtomicBool,
) -> Result<(), DataQueueProducerError>
where
    T: Send + 'static,
{
    if indices.is_empty() {
        return Ok(());
    }
    let prefetch = queue_capacity.max(worker_count).min(indices.len());
    thread::scope(|scope| {
        let (jobs_tx, jobs_rx) = bounded::<(usize, usize)>(prefetch);
        let (results_tx, results_rx) =
            bounded::<(usize, Result<T, DataQueueProducerError>)>(prefetch);
        for _ in 0..worker_count {
            let jobs = jobs_rx.clone();
            let results = results_tx.clone();
            let dataset = Arc::clone(dataset);
            scope.spawn(move || {
                while let Ok((sequence, index)) = jobs.recv() {
                    let value = match catch_unwind(AssertUnwindSafe(|| dataset.get(index))) {
                        Ok(Ok(value)) => Ok(value),
                        Ok(Err(message)) => Err(DataQueueProducerError::Dataset { index, message }),
                        Err(_) => Err(DataQueueProducerError::Panicked),
                    };
                    if results.send((sequence, value)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(results_tx);

        let mut scheduled = 0;
        while scheduled < prefetch {
            jobs_tx
                .send((scheduled, indices[scheduled]))
                .expect("scoped workers retain the job receiver");
            scheduled += 1;
        }
        let mut next = 0;
        let mut pending = BTreeMap::new();
        while next < indices.len() && !cancelled.load(Ordering::Acquire) {
            let (sequence, value) = results_rx
                .recv()
                .expect("scoped workers retain result senders");
            pending.insert(sequence, value);
            while let Some(value) = pending.remove(&next) {
                let value = value?;
                send_cancellable(output, value, cancelled)?;
                next += 1;
            }
            while scheduled < indices.len() && scheduled - next < prefetch {
                jobs_tx
                    .send((scheduled, indices[scheduled]))
                    .expect("scoped workers retain the job receiver");
                scheduled += 1;
            }
        }
        Ok(())
    })
}

fn send_cancellable<T>(
    sender: &Sender<T>,
    value: T,
    cancelled: &AtomicBool,
) -> Result<(), DataQueueProducerError> {
    send_with_retry(value, cancelled, |value| {
        sender.send_timeout(value, SEND_POLL_INTERVAL)
    })
}

// Separate transport from retry/cancellation policy so a full channel's timeout and
// subsequent state transitions can be tested without racing the OS thread scheduler.
fn send_with_retry<T>(
    mut value: T,
    cancelled: &AtomicBool,
    mut send: impl FnMut(T) -> Result<(), SendTimeoutError<T>>,
) -> Result<(), DataQueueProducerError> {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        match send(value) {
            Ok(()) => return Ok(()),
            Err(SendTimeoutError::Timeout(returned)) => value = returned,
            Err(SendTimeoutError::Disconnected(_)) => {
                return Err(DataQueueProducerError::ConsumerDisconnected);
            }
        }
    }
}

// Cleanup's before/after-join phases and Drop use the same drain operation. Keeping
// it shared lets the deterministic join-boundary test exercise actual drain behavior
// without depending on a late producer send racing a particular duplicated loop.
fn drain<T>(receiver: &Receiver<T>) {
    while receiver.try_recv().is_ok() {}
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
#[path = "../tests/support/data_queue_unit.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/support/data_queue_integration.rs"]
mod integration_tests;
