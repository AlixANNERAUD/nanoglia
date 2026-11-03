//! Greedy-by-size placement of activation tensors in the context arena, the strategy of
//! TFLite Micro's `GreedyMemoryPlanner`: tensors are placed from largest to smallest at the
//! lowest offset that does not overlap any already placed tensor alive at the same time.
//!
//! The planner works in caller-provided slices (carved from the graph block), so it allocates
//! nothing.

use nanoglia_core::align_up;

/// A tensor to place: its size and the range of operators during which it is alive (inclusive).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(crate) struct Interval {
    /// Size in bytes (0 for tensors that need no memory).
    pub size: u32,
    /// First operator index at which the tensor must exist.
    pub first: u32,
    /// Last operator index at which the tensor must exist.
    pub last: u32,
    /// Offset assigned by [`plan`].
    pub offset: u32,
}

impl Interval {
    fn overlaps_in_time(&self, other: &Interval) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// Assigns `offset` to every interval and returns the arena size, or `None` on overflow.
///
/// `order` is working memory of the same length as `intervals`.
pub(crate) fn plan(intervals: &mut [Interval], order: &mut [u32]) -> Option<usize> {
    debug_assert_eq!(intervals.len(), order.len());
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i as u32;
    }
    // Largest first; ties broken by index for determinism. Insertion sort: no allocation, and
    // graphs have at most a few hundred tensors.
    for i in 1..order.len() {
        let mut j = i;
        while j > 0 {
            let (a, b) = (order[j - 1] as usize, order[j] as usize);
            let swap = intervals[b].size > intervals[a].size
                || (intervals[b].size == intervals[a].size && b < a);
            if !swap {
                break;
            }
            order.swap(j - 1, j);
            j -= 1;
        }
    }

    let mut total = 0usize;
    for placed_count in 0..order.len() {
        let current = order[placed_count] as usize;
        let size = align_up(intervals[current].size as usize)?;
        if size == 0 {
            intervals[current].offset = 0;
            continue;
        }
        // Candidate offsets: 0 and the end of every conflicting placed tensor. Take the lowest
        // candidate that does not overlap any conflicting placed tensor.
        let mut best: Option<usize> = None;
        let candidates =
            core::iter::once(0usize).chain(order[..placed_count].iter().filter_map(|&p| {
                let p = &intervals[p as usize];
                (p.size != 0 && p.overlaps_in_time(&intervals[current]))
                    .then(|| align_up(p.offset as usize + p.size as usize))
                    .flatten()
            }));
        for candidate in candidates {
            if best.is_some_and(|b| b <= candidate) {
                continue;
            }
            let fits = order[..placed_count].iter().all(|&p| {
                let p = &intervals[p as usize];
                if p.size == 0 || !p.overlaps_in_time(&intervals[current]) {
                    return true;
                }
                let (start, end) = (p.offset as usize, p.offset as usize + p.size as usize);
                candidate + size <= start || candidate >= end
            });
            if fits {
                best = Some(candidate);
            }
        }
        let offset = best?;
        intervals[current].offset = u32::try_from(offset).ok()?;
        total = total.max(offset.checked_add(size)?);
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(size: u32, first: u32, last: u32) -> Interval {
        Interval {
            size,
            first,
            last,
            offset: u32::MAX,
        }
    }

    fn check(intervals: &[Interval], total: usize) {
        for (i, a) in intervals.iter().enumerate() {
            assert_eq!(a.offset as usize % nanoglia_core::ALIGN, 0);
            assert!(a.offset as usize + a.size as usize <= total);
            for b in &intervals[i + 1..] {
                if a.size == 0 || b.size == 0 || !a.overlaps_in_time(b) {
                    continue;
                }
                let disjoint = a.offset + a.size <= b.offset || b.offset + b.size <= a.offset;
                assert!(disjoint, "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn chain_reuses_memory() {
        // t0 -> op0 -> t1 -> op1 -> t2 -> op2 -> t3
        let mut v = [iv(100, 0, 0), iv(200, 0, 1), iv(100, 1, 2), iv(50, 2, 2)];
        let mut order = [0; 4];
        let total = plan(&mut v, &mut order).unwrap();
        check(&v, total);
        // Never more than two tensors alive: t1 (208) + t2 (112).
        assert!(total <= 208 + 112, "{total}");
    }

    #[test]
    fn all_alive_stacks() {
        let mut v = [iv(10, 0, 5), iv(20, 0, 5), iv(30, 0, 5)];
        let mut order = [0; 3];
        let total = plan(&mut v, &mut order).unwrap();
        check(&v, total);
        assert_eq!(total, 16 + 32 + 32);
    }

    #[test]
    fn pseudo_random_graphs_never_overlap() {
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for _ in 0..200 {
            let mut v = [Interval::default(); 24];
            for slot in v.iter_mut() {
                let first = next() % 20;
                *slot = iv(next() % 500, first, first + next() % 6);
            }
            let mut order = [0; 24];
            let total = plan(&mut v, &mut order).unwrap();
            check(&v, total);
        }
    }
}
