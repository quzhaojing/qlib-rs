use super::*;

#[test]
fn timeout_retry_preserves_the_unsent_value_and_observes_cancel_or_disconnect() {
    for outcome in ["success", "cancel", "disconnect"] {
        let (sender, receiver) = bounded(1);
        let original = Arc::new(vec![9]);
        sender.send(Arc::new(vec![7])).unwrap();
        let mut receiver = Some(receiver);
        let cancelled = AtomicBool::new(false);
        let mut attempts = 0;
        let result = send_with_retry(original.clone(), &cancelled, |value| {
            attempts += 1;
            assert!(
                Arc::ptr_eq(&value, &original),
                "retry must retain the exact unsent payload"
            );
            let result = sender.send_timeout(value, Duration::ZERO);
            if attempts == 1 {
                assert!(matches!(result, Err(SendTimeoutError::Timeout(_))));
                match outcome {
                    "success" => assert_eq!(*receiver.as_ref().unwrap().recv().unwrap(), vec![7]),
                    "cancel" => cancelled.store(true, Ordering::Release),
                    "disconnect" => drop(receiver.take()),
                    _ => unreachable!(),
                }
            }
            result
        });
        match outcome {
            "success" => {
                assert_eq!(result, Ok(()));
                assert_eq!(attempts, 2);
                assert!(Arc::ptr_eq(&receiver.unwrap().recv().unwrap(), &original));
            }
            "cancel" => {
                assert_eq!(result, Ok(()));
                assert_eq!(attempts, 1);
                assert_eq!(*receiver.unwrap().recv().unwrap(), vec![7]);
                assert_eq!(Arc::strong_count(&original), 1);
            }
            "disconnect" => {
                assert_eq!(result, Err(DataQueueProducerError::ConsumerDisconnected));
                assert_eq!(attempts, 2);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cleanup_drains_values_committed_while_joining_the_producer() {
    let mut queue = DataQueue::new(
        Arc::new(vec![1]),
        DataQueueConfig {
            queue_maxsize: 1,
            ..DataQueueConfig::default()
        },
    );
    queue.put(7).unwrap();
    let sender = queue.sender.as_ref().unwrap().clone();
    let receiver = queue.receiver.clone();
    let cancelled = Arc::clone(&queue.cancelled);
    let (release, released) = bounded(1);
    queue.producer = Some(thread::spawn(move || {
        released.recv().unwrap();
        // Model a send already committed before cancellation, completing during join.
        sender.send(9).unwrap();
    }));
    queue.cleanup_with_join(|producer| {
        assert!(cancelled.load(Ordering::Acquire));
        assert!(
            receiver.is_empty(),
            "the initial buffered value was drained"
        );
        release.send(()).unwrap();
        producer.join().unwrap();
        assert_eq!(
            receiver.len(),
            1,
            "the in-flight send finished before join returned"
        );
    });
    assert!(
        queue.receiver.is_empty(),
        "cleanup must also drain the late value"
    );
    assert!(queue.done());
    assert!(queue.producer.is_none());
    assert_eq!(queue.get(), Err(DataQueueError::Exhausted));
}

#[test]
fn private_channel_and_poison_recovery_paths_are_explicit() {
    let (sender, receiver) = bounded::<i32>(1);
    drop(receiver);
    assert_eq!(
        send_cancellable(&sender, 1, &AtomicBool::new(false)),
        Err(DataQueueProducerError::ConsumerDisconnected)
    );

    let (sender, _receiver) = bounded::<i32>(1);
    let cancelled = AtomicBool::new(true);
    assert_eq!(send_cancellable(&sender, 1, &cancelled), Ok(()));

    let (sender, receiver) = bounded::<i32>(1);
    sender.send(1).unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let worker = thread::spawn(move || send_cancellable(&sender, 2, &worker_cancelled));
    thread::sleep(Duration::from_millis(25));
    cancelled.store(true, Ordering::Release);
    assert_eq!(worker.join().unwrap(), Ok(()));
    assert_eq!(receiver.recv().unwrap(), 1);

    let mutex = Arc::new(Mutex::new(3));
    let poisoned = Arc::clone(&mutex);
    let _ = thread::spawn(move || {
        let _guard = poisoned.lock().unwrap();
        panic!("poison");
    })
    .join();
    assert_eq!(*lock_unpoisoned(&mutex), 3);

    let mut queue = DataQueue::new(Arc::new(vec![1]), DataQueueConfig::default());
    queue.sender.take();
    assert_eq!(queue.get(), Err(DataQueueError::Exhausted));

    let mut completed_with_value = DataQueue::new(Arc::new(vec![1]), DataQueueConfig::default());
    completed_with_value.put(5).unwrap();
    completed_with_value.done.store(true, Ordering::Release);
    assert_eq!(completed_with_value.get(), Ok(5));

    let mut missing_completion = DataQueue::new(Arc::new(vec![1]), DataQueueConfig::default());
    missing_completion.completion_sender.take();
    assert!(matches!(
        missing_completion.activate(),
        Err(DataQueueError::Closed)
    ));
    let mut missing_sampler = DataQueue::new(Arc::new(vec![1]), DataQueueConfig::default());
    missing_sampler.sampler.take();
    assert!(matches!(
        missing_sampler.activate(),
        Err(DataQueueError::Closed)
    ));

    {
        let queued = DataQueue::new(Arc::new(vec![1]), DataQueueConfig::default());
        queued.put(8).unwrap();
    }

    let dataset: Arc<dyn DataQueueDataset<i32>> = Arc::new(vec![9]);
    let (output, consumer) = bounded(1);
    drop(consumer);
    assert_eq!(
        produce_epoch_parallel(&dataset, &[0], 1, 1, &output, &AtomicBool::new(false)),
        Err(DataQueueProducerError::ConsumerDisconnected)
    );
}
