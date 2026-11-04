//! Strided copies: the output is written in order, and each of its elements is read from the
//! input at an offset that is an affine function of its coordinates. `TRANSPOSE`, `SLICE` and
//! `STRIDED_SLICE` are strided copies; they only move bytes, so one engine serves every element
//! type of the same size.

use crate::util::pod_struct;
use nanoglia_core::pod::{Pod, cast_slice, cast_slice_mut};
use nanoglia_core::{Detail, Error, MAX_RANK, Result};

pod_struct! {
    /// A strided copy, with the axes of the output outermost first.
    pub(crate) struct StridedCopy {
        /// Number of output elements along each axis, left-padded with ones.
        counts: [u32; MAX_RANK],
        /// Input element step along each output axis.
        steps: [i32; MAX_RANK],
        /// Input element offset of the first output element.
        start: u32,
        /// Number of input elements.
        in_len: u32,
        /// Number of output elements.
        out_len: u32,
        /// Element size in bytes: 1, 2, 4 or 8.
        size: u32,
    }
}

impl StridedCopy {
    /// The copy whose output axes are `axes`, each `(count, step)`, outermost first, starting at
    /// input element `start`, from an input of `in_len` elements of `size` bytes.
    ///
    /// Axes of one element are dropped and contiguous axes merged, so that the innermost loop is
    /// as long as possible. Fails with `error` if an element read would fall outside the input,
    /// or if a step or count does not fit the parameters.
    pub(crate) fn new(
        axes: &[(usize, i64)],
        start: i64,
        in_len: usize,
        size: usize,
        error: Error,
    ) -> Result<Self> {
        if !matches!(size, 1 | 2 | 4 | 8) || axes.len() > MAX_RANK {
            return Err(error);
        }
        let mut copy = Self {
            counts: [1; MAX_RANK],
            steps: [0; MAX_RANK],
            start: 0,
            in_len: u32::try_from(in_len).map_err(|_| error)?,
            out_len: 0,
            size: size as u32,
        };
        let mut out_len = 1usize;
        for &(count, _) in axes {
            out_len = out_len.checked_mul(count).ok_or(error)?;
        }
        if out_len == 0 {
            return Ok(copy);
        }
        copy.out_len = u32::try_from(out_len).map_err(|_| error)?;

        // Bounds of the input offsets read, and the normalized axes.
        let (mut lowest, mut highest) = (start, start);
        let mut merged = [(1usize, 0i64); MAX_RANK];
        let mut n = 0;
        for &(count, step) in axes {
            if count == 1 {
                continue;
            }
            let span = step.checked_mul(count as i64 - 1).ok_or(error)?;
            if span < 0 {
                lowest = lowest.checked_add(span).ok_or(error)?;
            } else {
                highest = highest.checked_add(span).ok_or(error)?;
            }
            // The previous (outer) axis continues this one: merge them.
            if n > 0 && merged[n - 1].1 == step.checked_mul(count as i64).ok_or(error)? {
                merged[n - 1] = (merged[n - 1].0 * count, step);
            } else {
                merged[n] = (count, step);
                n += 1;
            }
        }
        if lowest < 0 || highest >= in_len as i64 {
            return Err(error);
        }
        copy.start = start as u32;
        for (i, &(count, step)) in merged[..n].iter().enumerate() {
            let axis = MAX_RANK - n + i;
            copy.counts[axis] = u32::try_from(count).map_err(|_| error)?;
            copy.steps[axis] = i32::try_from(step).map_err(|_| error)?;
        }
        Ok(copy)
    }

    /// Number of output elements.
    pub(crate) fn out_len(&self) -> usize {
        self.out_len as usize
    }

    /// Copies from `input` to `output`, which must hold at least `in_len` and `out_len` elements.
    pub(crate) fn run(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        match self.size {
            1 => self.run_typed::<u8>(input, output),
            2 => self.run_typed::<u16>(input, output),
            4 => self.run_typed::<u32>(input, output),
            8 => self.run_typed::<u64>(input, output),
            _ => Err(Error::invalid_argument(Detail::None)),
        }
    }

    fn run_typed<T: Pod>(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let size = core::mem::size_of::<T>();
        let (in_len, out_len) = (self.in_len as usize, self.out_len as usize);
        let short = |need: usize, actual: usize| crate::util::size_mismatch(need, actual);
        let input = input
            .get(..in_len * size)
            .ok_or_else(|| short(in_len * size, input.len()))?;
        let output_len = output.len();
        let output = output
            .get_mut(..out_len * size)
            .ok_or_else(|| short(out_len * size, output_len))?;
        let input = cast_slice::<T>(input)?;
        let output = cast_slice_mut::<T>(output)?;
        if out_len == 0 {
            return Ok(());
        }
        // Parameters written by `new` keep every offset inside the input; `get` turns a
        // corrupted parameter block into an error rather than a panic.
        let out_of_bounds = || Error::invalid_argument(Detail::None);

        const LAST: usize = MAX_RANK - 1;
        let inner = self.counts[LAST] as usize;
        let step = self.steps[LAST] as isize;
        let mut index = [0u32; LAST];
        let mut base = i64::from(self.start);
        for row in output.chunks_exact_mut(inner) {
            let first = usize::try_from(base).map_err(|_| out_of_bounds())?;
            if step == 1 {
                let src = input.get(first..first + inner).ok_or_else(out_of_bounds)?;
                row.copy_from_slice(src);
            } else {
                let mut i = first;
                for value in row.iter_mut() {
                    *value = *input.get(i).ok_or_else(out_of_bounds)?;
                    i = i.wrapping_add_signed(step);
                }
            }
            for axis in (0..LAST).rev() {
                index[axis] += 1;
                base += i64::from(self.steps[axis]);
                if index[axis] < self.counts[axis] {
                    break;
                }
                base -= i64::from(self.steps[axis]) * i64::from(self.counts[axis]);
                index[axis] = 0;
            }
        }
        Ok(())
    }
}

/// The element strides of a row-major tensor of dimensions `dims`.
pub(crate) fn row_major_strides(dims: &[i32]) -> [i64; MAX_RANK] {
    let mut strides = [0i64; MAX_RANK];
    let mut stride = 1i64;
    for (slot, &dim) in strides.iter_mut().zip(dims).rev() {
        *slot = stride;
        stride = stride.saturating_mul(i64::from(dim.max(0)));
    }
    strides
}

/// Evaluates a kernel whose parameters are one [`StridedCopy`], from input 0 to output 0.
pub(crate) fn strided_eval(ctx: &mut nanoglia_core::backend::EvalCtx<'_, '_>) -> Result<()> {
    let copy: StridedCopy = *nanoglia_core::pod::read(ctx.persist)?;
    let input = ctx.inputs.first().copied().unwrap_or(&[]);
    let output: &mut [u8] = match ctx.outputs.first_mut() {
        Some(output) => output,
        None => return Err(crate::util::size_mismatch(copy.out_len(), 0)),
    };
    copy.run(input, output)
}
