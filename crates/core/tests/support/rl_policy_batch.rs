use super::*;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    length: usize,
    size: usize,
    merge_last: bool,
    groups: Vec<Vec<usize>>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[test]
fn ordered_boundaries_match_actual_batch_split_and_do_not_consume_rng() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_policy_batch.json")).unwrap();
    assert_eq!(fixture.cases.len(), 72);
    for case in fixture.cases {
        let mut rng = StdRng::seed_from_u64(93);
        let mut unused = rng.clone();
        let actual =
            minibatch_indices(case.length, case.size, false, case.merge_last, &mut rng).unwrap();
        assert_eq!(actual, case.groups);
        assert_eq!(rng.random::<u64>(), unused.random::<u64>());
    }
}

#[test]
fn shuffled_boundaries_keep_one_permutation_and_validate_sizes() {
    let mut rng = StdRng::seed_from_u64(12);
    let mut baseline = rng.clone();
    let expected = minibatch_indices(17, 17, true, false, &mut baseline)
        .unwrap()
        .remove(0);
    let actual = minibatch_indices(17, 3, true, true, &mut rng).unwrap();
    assert_eq!(
        actual.iter().map(Vec::len).collect::<Vec<_>>(),
        [3, 3, 3, 3, 5]
    );
    assert_eq!(actual.concat(), expected);
    assert_eq!(rng.random::<u64>(), baseline.random::<u64>());
    let mut sorted = actual.concat();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..17).collect::<Vec<_>>());
    assert_ne!(expected, sorted);
    for length in [0, 5] {
        let mut before = rng.clone();
        let error = minibatch_indices(length, 0, true, true, &mut rng).unwrap_err();
        assert_eq!(error, InvalidBatchSize);
        assert!(error.to_string().contains("positive"));
        assert_eq!(rng.random::<u64>(), before.random::<u64>());
    }
    assert_eq!(
        minibatch_indices(3, usize::MAX, false, true, &mut rng).unwrap(),
        vec![vec![0, 1, 2]]
    );
}
