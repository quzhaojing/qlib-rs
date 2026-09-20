//! Replay physical-index and scalar episode state shared by storage adapters.
//! ndarray owns flags; rand owns uniform replacement sampling. This is the source
//! metadata protocol, not a replacement tensor store or complete replay buffer.

use ndarray::Array1;
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ReplayIndexError {
    #[error("replay capacity is zero")]
    ZeroCapacity,
    #[error("replay done storage has not been initialized")]
    MissingDone,
    #[error("replay physical index is outside storage")]
    Index,
    #[error("replay stack count must be positive")]
    StackCount,
    #[error("cannot sample from an empty replay population")]
    EmptyPopulation,
    #[error("episode length exceeds native index range")]
    EpisodeLength,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayEpisode {
    pub index: usize,
    pub reward: f64,
    pub length: usize,
    pub start: usize,
}

#[derive(Clone, Debug)]
pub struct ReplayIndex {
    done: Array1<bool>,
    initialized: bool,
    index: usize,
    size: usize,
    last: usize,
    episode_reward: f64,
    episode_length: usize,
    episode_start: usize,
}

impl ReplayIndex {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            done: Array1::from_elem(capacity, false),
            initialized: false,
            index: 0,
            size: 0,
            last: 0,
            episode_reward: 0.,
            episode_length: 0,
            episode_start: 0,
        }
    }
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.done.len()
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    #[must_use]
    pub fn next_write_index(&self) -> usize {
        self.index
    }
    #[must_use]
    pub fn last_index(&self) -> usize {
        self.last
    }

    /// Marks the zero-filled done column as allocated by the storage backend.
    pub fn initialize_done(&mut self) {
        self.initialized = true;
    }

    /// Resets active positions, retaining the allocated done flags like source.
    /// Optional episode-statistics retention is independent of clearing length.
    pub fn reset(&mut self, keep_statistics: bool) {
        self.index = 0;
        self.size = 0;
        self.last = 0;
        if !keep_statistics {
            self.episode_reward = 0.;
            self.episode_length = 0;
            self.episode_start = 0;
        }
    }

    /// Source `_add_index`, called BEFORE data assignment. A storage error later
    /// must not roll this state back. `commit_done` is the separate assignment step.
    /// # Errors
    /// Capacity zero or native episode-length overflow; earlier state stays changed.
    pub fn advance(&mut self, reward: f64, done: bool) -> Result<ReplayEpisode, ReplayIndexError> {
        let index = self.index;
        self.last = index;
        self.size = self.size.saturating_add(1).min(self.capacity());
        if self.capacity() == 0 {
            return Err(ReplayIndexError::ZeroCapacity);
        }
        self.index = (self.index + 1) % self.capacity();
        self.episode_reward += reward;
        self.episode_length = self
            .episode_length
            .checked_add(1)
            .ok_or(ReplayIndexError::EpisodeLength)?;
        let result = ReplayEpisode {
            index,
            reward: if done {
                self.episode_reward
            } else {
                self.episode_reward * 0.
            },
            length: if done { self.episode_length } else { 0 },
            start: self.episode_start,
        };
        if done {
            self.episode_reward = 0.;
            self.episode_length = 0;
            self.episode_start = self.index;
        }
        Ok(result)
    }

    /// Commit the terminal flag when the storage adapter assigns that field.
    /// # Errors
    /// Rejects a physical index outside the allocated capacity.
    pub fn commit_done(&mut self, index: usize, done: bool) -> Result<(), ReplayIndexError> {
        *self.done.get_mut(index).ok_or(ReplayIndexError::Index)? = done;
        self.initialized = true;
        Ok(())
    }

    /// # Errors
    /// A never-initialized buffer has no done field, even when logically empty.
    pub fn unfinished_indices(&self) -> Result<Vec<usize>, ReplayIndexError> {
        if !self.initialized {
            return Err(ReplayIndexError::MissingDone);
        }
        if self.size == 0 {
            return Ok(Vec::new());
        }
        let last = (self.index + self.size - 1) % self.size;
        Ok(if self.done[last] {
            Vec::new()
        } else {
            vec![last]
        })
    }

    /// Previous positions stop at episode starts and the oldest retained item.
    /// This array-index boundary uses nonnegative physical positions.
    /// # Errors
    /// Returns missing done storage; empty active size follows `NumPy` array modulo-zero (0).
    pub fn previous(&self, indices: &[usize]) -> Result<Vec<usize>, ReplayIndexError> {
        if !self.initialized {
            return Err(ReplayIndexError::MissingDone);
        }
        Ok(indices
            .iter()
            .map(|&index| {
                if self.size == 0 {
                    return 0;
                }
                let previous = (index % self.size + self.size - 1) % self.size;
                (previous + usize::from(self.done[previous] || previous == self.last)) % self.size
            })
            .collect())
    }

    /// Next positions stop at terminal transitions and the last inserted item.
    /// # Errors
    /// Returns missing done storage or an invalid physical position.
    pub fn next(&self, indices: &[usize]) -> Result<Vec<usize>, ReplayIndexError> {
        if !self.initialized {
            return Err(ReplayIndexError::MissingDone);
        }
        indices
            .iter()
            .map(|&index| {
                let done = *self.done.get(index).ok_or(ReplayIndexError::Index)?;
                Ok(if self.size == 0 {
                    0
                } else {
                    (index + usize::from(!(done || index == self.last))) % self.size
                })
            })
            .collect()
    }

    /// Chronological all-sample order, or native uniform sampling with replacement.
    /// Frame availability follows source previous-index filtering, including its
    /// stack-count boundary. RNG is borrowed; zero/negative sizes never draw.
    /// # Errors
    /// Rejects stack count zero, missing initialized flags when filtering, and
    /// positive sampling from an empty population.
    pub fn sample_indices<R: Rng + ?Sized>(
        &self,
        size: i64,
        stack_count: usize,
        sample_available: bool,
        rng: &mut R,
    ) -> Result<Vec<usize>, ReplayIndexError> {
        if stack_count == 0 {
            return Err(ReplayIndexError::StackCount);
        }
        if size < 0 {
            return Ok(Vec::new());
        }
        if size > 0 && (stack_count == 1 || !sample_available) {
            return sample_population(self.size, size, rng);
        }
        let mut all: Vec<_> = (self.index..self.size).chain(0..self.index).collect();
        if stack_count > 1 && sample_available {
            let mut previous = all.clone();
            for _ in 0..stack_count - 2 {
                previous = self.previous(&previous)?;
            }
            let before = self.previous(&previous)?;
            all = all
                .into_iter()
                .zip(previous.into_iter().zip(before))
                .filter_map(|(index, (previous, before))| (previous != before).then_some(index))
                .collect();
        }
        if size == 0 {
            return Ok(all);
        }
        Ok(sample_population(all.len(), size, rng)?
            .into_iter()
            .map(|index| all[index])
            .collect())
    }
}

fn sample_population<R: Rng + ?Sized>(
    length: usize,
    size: i64,
    rng: &mut R,
) -> Result<Vec<usize>, ReplayIndexError> {
    let distribution = Uniform::new(0, length).map_err(|_| ReplayIndexError::EmptyPopulation)?;
    Ok((0..size).map(|_| distribution.sample(rng)).collect())
}

#[cfg(test)]
#[path = "../tests/support/rl_replay_index.rs"]
mod tests;
