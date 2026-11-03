//! The persistent buffer of the kernels: a parameter struct, optionally followed by one multiplier
//! and one shift per output channel and by further `i32` values.

use nanonn_core::pod::{Layout, Pod, Region};
use nanonn_core::quant::Multiplier;
use nanonn_core::{Detail, Error, Result};

/// Regions of a parameter struct `P`, of per-channel requantization arrays and of extra values.
pub(crate) struct ChannelLayout<P> {
    /// The parameter struct.
    pub params: Region<P>,
    /// One Q31 multiplier per output channel.
    pub multiplier: Region<i32>,
    /// One shift per output channel.
    pub shift: Region<i32>,
    /// Further `i32` values (kernel sums of fully-connected layers), possibly empty.
    pub extra: Region<i32>,
    /// Total size in bytes.
    pub size: usize,
}

impl<P: Pod> ChannelLayout<P> {
    /// The layout for `channels` output channels and `extra` further `i32` values.
    pub fn new(channels: usize, extra: usize) -> Self {
        let mut layout = Layout::new();
        let params = layout.push::<P>(1);
        let multiplier = layout.push::<i32>(channels);
        let shift = layout.push::<i32>(channels);
        let extra = layout.push::<i32>(extra);
        Self {
            params,
            multiplier,
            shift,
            extra,
            size: layout.size(),
        }
    }

    /// Writes the parameters and exactly one multiplier per channel in `persist`.
    pub fn write(
        &self,
        persist: &mut [u8],
        params: P,
        multipliers: impl Iterator<Item = Multiplier>,
    ) -> Result<()> {
        self.params.get_mut(persist)?[0] = params;
        let mut count = 0;
        for m in multipliers {
            if count >= self.multiplier.len() {
                return Err(inconsistent());
            }
            self.multiplier.get_mut(persist)?[count] = m.multiplier;
            self.shift.get_mut(persist)?[count] = m.shift;
            count += 1;
        }
        if count != self.multiplier.len() {
            return Err(inconsistent());
        }
        Ok(())
    }
}

/// The parameter struct at the start of a persistent buffer.
pub(crate) fn read_params<P: Pod>(persist: &[u8]) -> Result<P> {
    Ok(*nanonn_core::pod::read::<P>(persist)?)
}

/// Writes a parameter struct at the start of a persistent buffer.
pub(crate) fn write_params<P: Pod>(persist: &mut [u8], params: P) -> Result<()> {
    nanonn_core::pod::write(persist, params)
}

/// The persistent buffer does not match the parameters it holds.
pub(crate) fn inconsistent() -> Error {
    Error::invalid_argument(Detail::None)
}
