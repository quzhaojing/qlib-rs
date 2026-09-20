//! Object-index equivalence and stable factorization for `MultiIndex` level construction.
use super::{BuiltinFrameValue as B, TemporalFrameValue as T, TupleFrameValue as V};
use crate::RlCheckpointText;
use arrow_schema::{ArrowError, TimeUnit};
use indexmap::IndexMap;
use num_bigint::BigInt;
use num_traits::FromPrimitive;
use std::collections::HashSet;

// Keys model Pandas object-index equivalence, not Python hashes or Rust value
// equality. In particular bool/int/integral-float coalesce without f64 rounding,
// and NaNs compare equal here while distinct missing sentinel classes do not.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Key {
    Integer(BigInt),
    Float(u64),
    Nan,
    None,
    Na,
    Nat,
    Text(RlCheckpointText),
    Timestamp(i128, bool),
    Duration(i128),
    Tuple(Vec<Self>),
}

fn ticks(value: i64, unit: TimeUnit) -> Result<i128, ArrowError> {
    if value == i64::MIN {
        return Err(ArrowError::InvalidArgumentError(
            "temporal object uses reserved NaT ticks".into(),
        ));
    }
    let scale = match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    };
    Ok(i128::from(value) * scale)
}

pub(super) fn key(value: &V) -> Result<Key, ArrowError> {
    Ok(match value {
        V::PythonTemporal(value) => value.key()?,
        V::Tuple(values) => Key::Tuple(values.iter().map(key).collect::<Result<_, _>>()?),
        V::Scalar(T::Timestamp {
            ticks: value,
            unit,
            timezone,
        }) => Key::Timestamp(ticks(*value, *unit)?, timezone.is_some()),
        V::Scalar(T::Duration { ticks: value, unit }) => Key::Duration(ticks(*value, *unit)?),
        V::Scalar(T::Builtin(value)) => match value {
            B::Bool(value) => Key::Integer(BigInt::from(u8::from(*value))),
            B::Int(value) => Key::Integer(BigInt::from(*value)),
            B::UInt(value) => Key::Integer(BigInt::from(*value)),
            B::Float(value) if value.is_nan() => Key::Nan,
            B::Float(value) if value.is_finite() && value.fract() == 0.0 => {
                Key::Integer(BigInt::from_f64(*value).expect("finite integral float"))
            }
            B::Float(value) => Key::Float(value.to_bits()),
            B::None => Key::None,
            B::PandasNa => Key::Na,
            B::NotATime => Key::Nat,
            B::Text(value) => Key::Text(value.clone()),
        },
    })
}

/// Stable object factorization with the first nonmissing representative retained.
/// `None` codes correspond to Python's -1 when missing values use a sentinel.
/// Uniques remain object cells; dtype inference and category sorting are separate.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectFactorization {
    pub codes: Vec<Option<usize>>,
    pub uniques: Vec<V>,
}

/// Check object-index uniqueness without conflating different missing sentinels.
/// Numeric equality is exact across bool/i64/u64/f64; timestamp equality compares
/// absolute ticks and awareness, and tuple equality applies recursively.
/// # Errors
/// Rejects reserved temporal ticks anywhere in the input, even after duplicates.
pub fn object_index_is_unique(values: &[V]) -> Result<bool, ArrowError> {
    let keys = values.iter().map(key).collect::<Result<Vec<_>, _>>()?;
    Ok(keys.iter().collect::<HashSet<_>>().len() == keys.len())
}

/// Factorize object cells in encounter order (`sort=False`).
/// Top-level missing values either receive no code or coalesce to a NaN unique.
/// Missing values nested inside tuples retain their distinct sentinel classes.
/// # Errors
/// Rejects reserved temporal ticks before publishing any partial result.
pub fn factorize_index_objects(
    values: &[V],
    use_na_sentinel: bool,
) -> Result<ObjectFactorization, ArrowError> {
    factorize_with_keys(values, use_na_sentinel).map(|(result, _)| result)
}

// Keep keys alongside their representatives so sorted factoring does not rebuild
// and revalidate every recursive key after successful stable factoring.
pub(super) fn factorize_with_keys(
    values: &[V],
    use_na_sentinel: bool,
) -> Result<(ObjectFactorization, Vec<Key>), ArrowError> {
    let keys = values.iter().map(key).collect::<Result<Vec<_>, _>>()?;
    let mut lookup = IndexMap::new();
    let mut uniques = vec![];
    let mut codes = Vec::with_capacity(values.len());
    for (value, mut key) in values.iter().zip(keys) {
        let missing = matches!(key, Key::None | Key::Nan | Key::Na | Key::Nat);
        if missing && use_na_sentinel {
            codes.push(None);
            continue;
        }
        if missing {
            key = Key::Nan;
        }
        let next = lookup.len();
        let code = *lookup.entry(key).or_insert_with(|| {
            uniques.push(if missing {
                V::Scalar(T::Builtin(B::Float(f64::NAN)))
            } else {
                value.clone()
            });
            next
        });
        codes.push(Some(code));
    }
    Ok((
        ObjectFactorization { codes, uniques },
        lookup.into_keys().collect(),
    ))
}

#[cfg(test)]
#[path = "object_factorization_tests.rs"]
mod tests;
