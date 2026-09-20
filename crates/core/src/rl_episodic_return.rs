//! Source episodic returns and generalized advantage estimation for RL policies.
//! Arrays/arithmetic use ndarray; the reverse recurrence is policy-specific.

use ndarray::{Array1, ArrayView1, Zip};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Clone)]
pub struct EpisodicReturnInput<'a> {
    pub rewards: ArrayView1<'a, f64>,
    pub terminated: ArrayView1<'a, bool>,
    pub truncated: ArrayView1<'a, bool>,
    /// Source buffer's !terminated at the selected indices, independently of
    /// batch end flags. Only inspected when `next_values` is present.
    pub bootstrap_valid: ArrayView1<'a, bool>,
    pub indices: &'a [usize],
    pub unfinished_indices: &'a [usize],
    pub next_values: Option<ArrayView1<'a, f64>>,
    pub values: Option<ArrayView1<'a, f64>>,
}

#[derive(Debug)]
pub struct EpisodicReturns {
    pub returns: Array1<f64>,
    pub advantages: Array1<f64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EpisodicReturnError {
    #[error("{field} length {actual} does not match rewards length {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("missing next-state values requires GAE lambda close to one")]
    MissingBootstrap,
}

fn length(field: &'static str, actual: usize, expected: usize) -> Result<(), EpisodicReturnError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EpisodicReturnError::Length {
            field,
            actual,
            expected,
        })
    }
}

/// Mirrors Tianshou's `compute_episodic_return` boundary on prepared dense arrays.
/// Preserve input order; indices identify unfinished buffer entries, not sorting.
/// F32 predictions may be losslessly widened before this F64 recurrence. Source
/// returns/advantages are F64; policy-specific normalization/casting is separate.
///
/// # Errors
/// Rejects inconsistent vector lengths and absent bootstrap values with lambda
/// not close to one. Like the source static helper, does not enforce the separate
/// policy constructor's gamma/lambda range checks or sanitize nonfinite values.
pub fn episodic_returns(
    input: &EpisodicReturnInput<'_>,
    gamma: f64,
    gae_lambda: f64,
) -> Result<EpisodicReturns, EpisodicReturnError> {
    let count = input.rewards.len();
    length("terminated", input.terminated.len(), count)?;
    length("truncated", input.truncated.len(), count)?;
    length("indices", input.indices.len(), count)?;
    let next_values = if let Some(next) = input.next_values {
        length("next_values", next.len(), count)?;
        length("bootstrap_valid", input.bootstrap_valid.len(), count)?;
        Zip::from(next)
            .and(input.bootstrap_valid)
            .map_collect(|&value, &valid| value * f64::from(u8::from(valid)))
    } else {
        if !crate::exchange_trade::numpy_is_close(gae_lambda, 1.) {
            return Err(EpisodicReturnError::MissingBootstrap);
        }
        Array1::zeros(count)
    };
    let values = if let Some(values) = input.values {
        length("values", values.len(), count)?;
        values.to_owned()
    } else {
        // np.roll wraps the last masked next value to the first position; it is
        // not reset at episode boundaries. Vec also handles an empty input.
        let mut rolled = next_values.to_vec();
        if !rolled.is_empty() {
            rolled.rotate_right(1);
        }
        Array1::from_vec(rolled)
    };
    let delta = Zip::from(input.rewards)
        .and(&next_values)
        .and(&values)
        .map_collect(|&reward, &next, &value| reward + next * gamma - value);
    let unfinished: HashSet<_> = input.unfinished_indices.iter().copied().collect();
    let mut advantages = Array1::zeros(count);
    let mut gae = 0.;
    for index in (0..count).rev() {
        let end = input.terminated[index]
            || input.truncated[index]
            || unfinished.contains(&input.indices[index]);
        // Keep multiplication even at an episode boundary: 0 * NaN is NaN in
        // the source. Neither terminal masks nor zero gamma sanitize NaN/Inf.
        let discount = f64::from(u8::from(!end)) * (gamma * gae_lambda);
        gae = delta[index] + discount * gae;
        advantages[index] = gae;
    }
    let returns = &advantages + &values;
    Ok(EpisodicReturns {
        returns,
        advantages,
    })
}

#[cfg(test)]
#[path = "../tests/support/rl_episodic_return.rs"]
mod tests;
