//! `NumPy` 2.4.0 object-index argsort compatibility, not a general sorting API.
// Adapted from NumPy's npy_aquicksort_impl / npy_aheapsort (Charles R. Harris
// and NumPy contributors). BSD-3-Clause; see crates/core/NUMPY-LICENSE.txt.
// Object comparisons may be non-total or fail. Rust's standard sort requires
// a total order and changes observable ties/comparison order for these inputs.
use std::cmp::Ordering;

type Compare<'a> = dyn FnMut(usize, usize) -> Option<Ordering> + 'a;

fn less(compare: &mut Compare<'_>, left: usize, right: usize) -> Option<bool> {
    Some(compare(left, right)? == Ordering::Less)
}

fn sift(
    order: &mut [usize],
    mut hole: usize,
    value: usize,
    compare: &mut Compare<'_>,
) -> Option<()> {
    let mut child = hole * 2 + 1;
    while child < order.len() {
        if child + 1 < order.len() && less(compare, order[child], order[child + 1])? {
            child += 1;
        }
        if !less(compare, value, order[child])? {
            break;
        }
        order[hole] = order[child];
        hole = child;
        child = hole * 2 + 1;
    }
    order[hole] = value;
    Some(())
}

fn heap(order: &mut [usize], compare: &mut Compare<'_>) -> Option<()> {
    for root in (0..order.len() / 2).rev() {
        sift(order, root, order[root], compare)?;
    }
    for end in (1..order.len()).rev() {
        order.swap(0, end);
        let value = order[0];
        sift(&mut order[..end], 0, value, compare)?;
    }
    Some(())
}

pub(super) fn sort(order: &mut [usize], compare: &mut Compare<'_>) -> Option<()> {
    if order.is_empty() {
        return Some(());
    }
    let limit = i32::try_from(order.len().ilog2() * 2).expect("usize logarithm fits i32");
    let mut pending = vec![(0, order.len() - 1, limit)];
    while let Some((mut low, mut high, mut depth)) = pending.pop() {
        if depth < 0 {
            heap(&mut order[low..=high], compare)?;
            continue;
        }
        while high - low > 15 {
            let middle = low + (high - low) / 2;
            if less(compare, order[middle], order[low])? {
                order.swap(middle, low);
            }
            if less(compare, order[high], order[middle])? {
                order.swap(high, middle);
            }
            if less(compare, order[middle], order[low])? {
                order.swap(middle, low);
            }
            let pivot = order[middle];
            let mut left = low;
            let mut right = high - 1;
            order.swap(middle, right);
            loop {
                left += 1;
                while less(compare, order[left], pivot)? && left < right {
                    left += 1;
                }
                right -= 1;
                while less(compare, pivot, order[right])? && left < right {
                    right -= 1;
                }
                if left >= right {
                    break;
                }
                order.swap(left, right);
            }
            order.swap(left, high - 1);
            depth -= 1;
            if left - low < high - left {
                pending.push((left + 1, high, depth));
                high = left - 1;
            } else {
                pending.push((low, left - 1, depth));
                low = left + 1;
            }
        }
        for position in low + 1..=high {
            let value = order[position];
            let mut hole = position;
            while hole > low && less(compare, value, order[hole - 1])? {
                order[hole] = order[hole - 1];
                hole -= 1;
            }
            order[hole] = value;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adversarial_partition_depth_and_comparator_failures_preserve_permutations() {
        for len in [256, 1024, 4096] {
            let values = (0..len / 2)
                .step_by(2)
                .flat_map(|i| [i, i + len / 2])
                .chain((1..len).step_by(2))
                .collect::<Vec<_>>();
            assert_eq!(values.len(), len);
            let mut order = (0..len).collect::<Vec<_>>();
            let mut calls = 0;
            assert_eq!(
                sort(&mut order, &mut |l, r| {
                    calls += 1;
                    Some(values[l].cmp(&values[r]))
                }),
                Some(())
            );
            assert_eq!(
                order.iter().map(|&i| values[i]).collect::<Vec<_>>(),
                (0..len).collect::<Vec<_>>()
            );
            let stride = if len == 256 { 1 } else { 127 };
            for fail_at in (0..calls).step_by(stride) {
                let mut working = (0..len).collect::<Vec<_>>();
                let mut count = 0;
                assert_eq!(
                    sort(&mut working, &mut |l, r| {
                        let fail = count == fail_at;
                        count += 1;
                        (!fail).then(|| values[l].cmp(&values[r]))
                    }),
                    None
                );
                assert_eq!(count, fail_at + 1);
            }
        }
        // Deliberately inconsistent comparator: the adapter must terminate and
        // retain all indices even when both a<b and b<a are reported true.
        let mut order = (0..65).collect::<Vec<_>>();
        assert_eq!(sort(&mut order, &mut |_, _| Some(Ordering::Less)), Some(()));
        order.sort_unstable();
        assert_eq!(order, (0..65).collect::<Vec<_>>());
    }

    #[test]
    fn organ_pipe_depth_fallback_propagates_comparison_errors() {
        let values = (0..2048).chain((0..2048).rev()).collect::<Vec<_>>();
        let mut order = (0..values.len()).collect::<Vec<_>>();
        let mut calls = 0;
        assert_eq!(
            sort(&mut order, &mut |l, r| {
                calls += 1;
                Some(values[l].cmp(&values[r]))
            }),
            Some(())
        );
        let mut expected = values.clone();
        expected.sort_unstable();
        assert_eq!(
            order.iter().map(|&i| values[i]).collect::<Vec<_>>(),
            expected
        );
        for fail_at in (0..calls).step_by(127) {
            let mut working = (0..values.len()).collect::<Vec<_>>();
            let mut count = 0;
            assert_eq!(
                sort(&mut working, &mut |l, r| {
                    let fail = count == fail_at;
                    count += 1;
                    (!fail).then(|| values[l].cmp(&values[r]))
                }),
                None
            );
            assert_eq!(count, fail_at + 1);
        }
    }

    #[test]
    fn partition_heap_and_every_comparison_failure_are_explicit() {
        for len in 0..132 {
            let values = (0..len).map(|i| (i * 53) % 137).collect::<Vec<_>>();
            let mut expected = (0..len).collect::<Vec<_>>();
            expected.sort_by_key(|&i| values[i]);
            let mut quick = (0..len).collect::<Vec<_>>();
            assert_eq!(
                sort(&mut quick, &mut |l, r| Some(values[l].cmp(&values[r]))),
                Some(())
            );
            assert_eq!(quick, expected);
            let mut heaps = (0..len).rev().collect::<Vec<_>>();
            assert_eq!(
                heap(&mut heaps, &mut |l, r| Some(values[l].cmp(&values[r]))),
                Some(())
            );
            assert_eq!(heaps, expected);
        }
        for engine in [sort, heap] {
            let values = (0..65).map(|i| (i * 17) % 67).collect::<Vec<_>>();
            let mut count = 0;
            let mut order = (0..values.len()).collect::<Vec<_>>();
            assert_eq!(
                engine(&mut order, &mut |l, r| {
                    count += 1;
                    Some(values[l].cmp(&values[r]))
                }),
                Some(())
            );
            for fail_at in 0..count {
                let mut calls = 0;
                let mut working = (0..values.len()).collect::<Vec<_>>();
                assert_eq!(
                    engine(&mut working, &mut |l, r| {
                        let fail = calls == fail_at;
                        calls += 1;
                        (!fail).then(|| values[l].cmp(&values[r]))
                    }),
                    None
                );
                assert_eq!(calls, fail_at + 1);
            }
        }
    }
}
