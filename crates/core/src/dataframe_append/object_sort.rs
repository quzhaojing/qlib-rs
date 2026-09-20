//! Object-category ordering and encounter-order fallback for `MultiIndex` levels.
use super::object_factorization::{Key, factorize_with_keys};
use super::{ObjectFactorization, TupleFrameValue};
use arrow_schema::ArrowError;
use num_bigint::BigInt;
use num_traits::FromPrimitive;
use std::cmp::Ordering;

fn integer_float(left: &BigInt, right: f64) -> Ordering {
    if let Some(integer) = BigInt::from_f64(right) {
        left.cmp(&integer).then_with(|| {
            if right.fract() > 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
    } else if right.is_sign_positive() {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

fn compare(left: &Key, right: &Key) -> Option<Ordering> {
    use Key::{Duration, Float, Integer, Nan, Nat, Text, Timestamp, Tuple};
    if left == right {
        return Some(Ordering::Equal);
    }
    match (left, right) {
        (Integer(l), Integer(r)) => Some(l.cmp(r)),
        (Float(l), Float(r)) => f64::from_bits(*l).partial_cmp(&f64::from_bits(*r)),
        (Integer(l), Float(r)) => Some(integer_float(l, f64::from_bits(*r))),
        (Float(l), Integer(r)) => Some(integer_float(r, f64::from_bits(*l)).reverse()),
        (Text(l), Text(r)) => Some(l.as_code_points().cmp(r.as_code_points())),
        (Timestamp(l, la), Timestamp(r, ra)) if la == ra => Some(l.cmp(r)),
        (Duration(l), Duration(r)) => Some(l.cmp(r)),
        (Nan, Integer(_) | Float(_) | Nan)
        | (Integer(_) | Float(_), Nan)
        | (Nat, Timestamp(..) | Duration(_) | Nat)
        | (Timestamp(..) | Duration(_), Nat) => Some(Ordering::Equal),
        (Tuple(l), Tuple(r)) => {
            for (l, r) in l.iter().zip(r) {
                if l != r {
                    return compare(l, r);
                }
            }
            Some(l.len().cmp(&r.len()))
        }
        _ => None,
    }
}

fn direct(keys: &[Key], selected: Vec<usize>) -> Option<Vec<usize>> {
    let mut order = selected;
    super::object_argsort::sort(&mut order, &mut |l, r| compare(&keys[l], &keys[r]))?;
    Some(order)
}

fn missing(key: &Key) -> bool {
    matches!(key, Key::None | Key::Na | Key::Nat | Key::Nan)
}

fn mixed(keys: &[Key]) -> Option<Vec<usize>> {
    let mut numbers = Vec::new();
    let mut strings = Vec::new();
    let mut nulls = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if matches!(key, Key::Text(_)) {
            strings.push(i);
        } else if missing(key) {
            nulls.push(i);
        } else {
            numbers.push(i);
        }
    }
    let mut result = direct(keys, numbers)?;
    // The selected keys are exclusively text: code-point comparison is total.
    // Retain the same argsort (including tie order), while making its invariant
    // distinct from the genuinely fallible mixed numeric comparison above.
    result.extend(direct(keys, strings).expect("text keys compare totally"));
    result.extend(nulls);
    Some(result)
}

fn tuples(keys: &[Key]) -> Option<Vec<usize>> {
    let rows = keys
        .iter()
        .map(|v| match v {
            Key::Tuple(values) => Some(values.clone()),
            Key::Text(value) => Some(
                value
                    .as_code_points()
                    .iter()
                    .map(|&point| {
                        Key::Text(
                            crate::RlCheckpointText::try_from_code_points([point])
                                .expect("existing code point"),
                        )
                    })
                    .collect(),
            ),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return None;
    }
    let mut row_ranks = vec![Vec::new(); rows.len()];
    for column in 0..width {
        let cells = rows
            .iter()
            .map(|v| v.get(column).cloned().unwrap_or(Key::None))
            .collect::<Vec<_>>();
        // lexsort_indexer categoricals place every missing sentinel last.
        let present = cells
            .iter()
            .enumerate()
            .filter(|(_, k)| !missing(k))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let present_keys = present
            .iter()
            .map(|&i| cells[i].clone())
            .collect::<Vec<_>>();
        let order = sorted(&present_keys)?;
        let mut ranks = vec![usize::MAX; rows.len()];
        let mut rank = 0;
        for (position, &index) in order.iter().enumerate() {
            if position > 0 && present_keys[index] != present_keys[order[position - 1]] {
                rank += 1;
            }
            ranks[present[index]] = rank;
        }
        for (row, rank) in row_ranks.iter_mut().zip(ranks) {
            row.push(rank);
        }
    }
    let mut order = (0..keys.len()).collect::<Vec<_>>();
    order.sort_by(|&l, &r| row_ranks[l].cmp(&row_ranks[r]));
    Some(order)
}

fn sorted(keys: &[Key]) -> Option<Vec<usize>> {
    direct(keys, (0..keys.len()).collect()).or_else(|| {
        if matches!(keys.first(), Some(Key::Tuple(_))) {
            tuples(keys)
        } else {
            mixed(keys)
        }
    })
}

/// Factorize object levels, sorting when comparisons permit and retaining the
/// original encounter order when categorical ordering fails. Missing cells use
/// `None` codes. The returned representatives remain object cells; dtype
/// reconstruction is a separate boundary.
/// # Errors
/// Rejects invalid temporal payloads without publishing partial codes.
pub fn factorize_level_objects(
    values: &[TupleFrameValue],
) -> Result<ObjectFactorization, ArrowError> {
    let (mut result, keys) = factorize_with_keys(values, true)?;
    if let Some(order) = sorted(&keys) {
        let mut reverse = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            reverse[old] = new;
        }
        for code in result.codes.iter_mut().flatten() {
            *code = reverse[*code];
        }
        result.uniques = order
            .into_iter()
            .map(|old| result.uniques[old].clone())
            .collect();
    }
    Ok(result)
}

#[cfg(test)]
#[path = "object_sort_tests.rs"]
mod tests;
