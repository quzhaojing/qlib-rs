use super::*;
use rand::{RngCore, SeedableRng, rngs::StdRng};

#[derive(Deserialize)]
struct Step {
    reward: f64,
    done: bool,
    added: ReplayEpisode,
    size: usize,
    next_write: usize,
    last: usize,
    unfinished: Vec<usize>,
    previous: Vec<usize>,
    following: Vec<usize>,
    available: Vec<usize>,
}
#[derive(Deserialize)]
struct Case {
    capacity: usize,
    stack: usize,
    available: bool,
    steps: Vec<Step>,
}

#[test]
fn state_and_frame_availability_match_216_real_replay_additions() {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_replay_index.json")).unwrap();
    assert_eq!(fixture.cases.len(), 24);
    for case in fixture.cases {
        let mut index = ReplayIndex::new(case.capacity);
        let mut rng = StdRng::seed_from_u64(21);
        let mut untouched_rng = rng.clone();
        assert!(index.is_empty());
        assert_eq!(index.capacity(), case.capacity);
        for step in case.steps {
            let added = index.advance(step.reward, step.done).unwrap();
            index.commit_done(added.index, step.done).unwrap();
            assert_eq!(added.index, step.added.index);
            assert!((added.reward - step.added.reward).abs() < 1e-14);
            assert_eq!(added.length, step.added.length);
            assert_eq!(added.start, step.added.start);
            assert_eq!(index.len(), step.size);
            assert_eq!(index.next_write_index(), step.next_write);
            assert_eq!(index.last_index(), step.last);
            assert_eq!(index.unfinished_indices().unwrap(), step.unfinished);
            let slots: Vec<_> = (0..case.capacity).collect();
            assert_eq!(index.previous(&slots).unwrap(), step.previous);
            assert_eq!(index.next(&slots).unwrap(), step.following);
            assert_eq!(
                index
                    .sample_indices(0, case.stack, case.available, &mut rng)
                    .unwrap(),
                step.available
            );
        }
        assert_eq!(rng.next_u64(), untouched_rng.next_u64());
    }
}

#[test]
fn uniform_sampling_replaces_and_respects_available_frames_and_rng() {
    let mut index = ReplayIndex::new(5);
    for _ in 0..5 {
        let added = index.advance(1., false).unwrap();
        index.commit_done(added.index, false).unwrap();
    }
    let mut rng = StdRng::seed_from_u64(95);
    let mut replay_rng = rng.clone();
    let samples = index.sample_indices(30_000, 1, false, &mut rng).unwrap();
    assert_eq!(
        samples,
        index
            .sample_indices(30_000, 1, false, &mut replay_rng)
            .unwrap()
    );
    let mut counts = [0; 5];
    for sample in samples {
        counts[sample] += 1;
    }
    for count in counts {
        assert!((count - 6000_i32).abs() < 416);
    }
    assert_eq!(rng.next_u64(), replay_rng.next_u64());
    assert_eq!(
        index.sample_indices(20, 5, false, &mut rng).unwrap(),
        index.sample_indices(20, 1, false, &mut replay_rng).unwrap()
    );
    let available = index.sample_indices(0, 3, true, &mut rng).unwrap();
    assert_eq!(available, [2, 3, 4]);
    let filtered = index.sample_indices(40, 3, true, &mut rng).unwrap();
    assert_eq!(filtered.len(), 40);
    assert!(filtered.iter().all(|index| available.contains(index)));
    assert_eq!(
        index.sample_indices(0, 7, true, &mut rng).unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(
        index.sample_indices(1, 7, true, &mut rng),
        Err(ReplayIndexError::EmptyPopulation)
    );
    assert_eq!(
        index.sample_indices(-1, 7, true, &mut rng).unwrap(),
        Vec::<usize>::new()
    );
}

#[test]
fn fresh_reset_partial_assignment_and_error_boundaries_preserve_state() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut index = ReplayIndex::new(3);
    assert_eq!(
        index.unfinished_indices(),
        Err(ReplayIndexError::MissingDone)
    );
    assert_eq!(index.previous(&[]), Err(ReplayIndexError::MissingDone));
    assert_eq!(index.next(&[]), Err(ReplayIndexError::MissingDone));
    assert_eq!(
        index.sample_indices(0, 1, false, &mut rng).unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(
        index.sample_indices(1, 1, false, &mut rng),
        Err(ReplayIndexError::EmptyPopulation)
    );
    assert_eq!(
        index.sample_indices(0, 2, true, &mut rng),
        Err(ReplayIndexError::MissingDone)
    );
    assert_eq!(
        index.sample_indices(0, 3, true, &mut rng),
        Err(ReplayIndexError::MissingDone)
    );
    assert_eq!(
        index.sample_indices(-1, 0, false, &mut rng),
        Err(ReplayIndexError::StackCount)
    );
    assert_eq!(index.commit_done(3, false), Err(ReplayIndexError::Index));
    let result = index.advance(2., false).unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(
        index.unfinished_indices(),
        Err(ReplayIndexError::MissingDone)
    );
    index.commit_done(result.index, false).unwrap();
    assert_eq!(index.next(&[3]), Err(ReplayIndexError::Index));
    index.reset(true);
    assert!(index.is_empty());
    assert_eq!(index.unfinished_indices().unwrap(), Vec::<usize>::new());
    assert_eq!(index.previous(&[0, 1]).unwrap(), [0, 0]);
    assert_eq!(index.next(&[0, 1]).unwrap(), [0, 0]);
    let result = index.advance(3., true).unwrap();
    assert_eq!(result.length, 2);
    assert_eq!(result.reward.to_bits(), 5_f64.to_bits());
    index.reset(false);
    let result = index.advance(3., true).unwrap();
    assert_eq!(result.length, 1);
    assert_eq!(result.reward.to_bits(), 3_f64.to_bits());
    let mut zero = ReplayIndex::new(0);
    assert_eq!(zero.advance(1., false), Err(ReplayIndexError::ZeroCapacity));
    assert_eq!(zero.len(), 0);
    assert_eq!(zero.episode_length, 0);
    index.episode_length = usize::MAX;
    assert_eq!(
        index.advance(1., false),
        Err(ReplayIndexError::EpisodeLength)
    );
}

#[test]
fn unfinished_nonfinite_rewards_do_not_become_finite_zero() {
    for reward in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut index = ReplayIndex::new(1);
        assert!(index.advance(reward, false).unwrap().reward.is_nan());
    }
    let mut index = ReplayIndex::new(1);
    assert_eq!(
        index.advance(-1., false).unwrap().reward.to_bits(),
        (-0_f64).to_bits()
    );
}
