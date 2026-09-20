//! Source minibatch boundaries over native batch positions.

use rand::{Rng, seq::SliceRandom};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("policy minibatch size must be positive")]
pub struct InvalidBatchSize;

/// Split batch positions like Tianshou's `Batch.split`. Shuffle once per call
/// with the caller-owned RNG. With `merge_last`, append an incomplete final chunk
/// to the preceding one, which may therefore contain up to 2 * size - 1 entries.
/// Native seeded permutations are reproducible, not bitwise `NumPy` RNG replay.
/// # Errors
/// Rejects size zero, including for empty batches.
pub fn minibatch_indices<R: Rng + ?Sized>(
    length: usize,
    size: usize,
    shuffle: bool,
    merge_last: bool,
    rng: &mut R,
) -> Result<Vec<Vec<usize>>, InvalidBatchSize> {
    if size == 0 {
        return Err(InvalidBatchSize);
    }
    let mut indices: Vec<_> = (0..length).collect();
    if shuffle {
        indices.shuffle(rng);
    }
    let mut batches: Vec<_> = indices.chunks(size).map(<[usize]>::to_vec).collect();
    if merge_last && length % size > 0 && length > size {
        // The length check guarantees two batches. Preserve the order within
        // the generated permutation when joining them.
        let tail = batches.split_off(batches.len() - 2);
        batches.push(tail.concat());
    }
    Ok(batches)
}

#[cfg(test)]
#[path = "../tests/support/rl_policy_batch.rs"]
mod tests;
