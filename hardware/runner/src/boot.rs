//! The byte that survives a reset: how the previous run ended, and how many runs in a row ended in
//! a crash before the host said a word.
//!
//! A firmware that crashes while it starts would reset forever, out of reach of the host. So the
//! handler of every crash counts it, in [`crash`], before it resets: at [`MAX_FAILED_BOOTS`] it
//! resets into the bootloader instead, where a new firmware can be flashed. The count is made
//! there, not when the firmware starts, so that a crash in the very first instructions of `main`
//! counts too. A request of the host clears it ([`contact`]).
//!
//! | Bits | Field |
//! | --- | --- |
//! | 7..4 | [`MARK`], or the byte means nothing (0 after a power-on) |
//! | 3..2 | how the previous run ended, a [`Crash`] |
//! | 1..0 | crashes in a row without a request of the host |

use crate::board::{Board, Reset};

/// High half of a valid byte.
pub const MARK: u8 = 0xA0;
/// Crashes in a row, without a request of the host in between, after which the firmware resets
/// into the bootloader.
pub const MAX_FAILED_BOOTS: u8 = 3;

/// How a run ended, when not by a reset the host asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Crash {
    /// It did not crash.
    None = 0,
    /// A Rust panic.
    Panic = 1,
    /// A processor fault.
    Fault = 2,
    /// The watchdog: nothing moved for too long.
    Watchdog = 3,
}

impl Crash {
    fn from_bits(bits: u8) -> Self {
        match bits & 3 {
            1 => Self::Panic,
            2 => Self::Fault,
            3 => Self::Watchdog,
            _ => Self::None,
        }
    }
}

/// The crash and the count of failed boots in `retained`.
pub fn decode(retained: u8) -> (Crash, u8) {
    if retained & 0xF0 != MARK {
        return (Crash::None, 0);
    }
    (Crash::from_bits(retained >> 2), retained & 3)
}

/// The retained byte of `crash` and `failed` failed boots.
pub fn encode(crash: Crash, failed: u8) -> u8 {
    MARK | (crash as u8) << 2 | failed.min(3)
}

/// What a crash handler does after a crash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// The retained byte to write before the reset.
    pub retained: u8,
    /// Where to reset: the bootloader after [`MAX_FAILED_BOOTS`] crashes in a row.
    pub to: Reset,
}

/// Counts the crash `kind` in `retained`, the byte as the handler finds it. Every handler of a
/// crash (panic, fault, watchdog) calls this, writes [`Verdict::retained`] and resets to
/// [`Verdict::to`].
pub fn crash(retained: u8, kind: Crash) -> Verdict {
    let failed = decode(retained).1 + 1;
    if failed >= MAX_FAILED_BOOTS {
        // The count starts again: the bootloader runs this firmware if none replaces it.
        return Verdict {
            retained: encode(Crash::None, 0),
            to: Reset::Bootloader,
        };
    }
    Verdict {
        retained: encode(kind, failed),
        to: Reset::Application,
    }
}

/// What the retained byte says when the firmware starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boot {
    /// How the previous run ended.
    pub crash: Crash,
    /// Crashes in a row without a request of the host, the one of the previous run included.
    pub failed: u8,
}

/// Reads the retained byte when the runner starts: how the previous run ended, which is forgotten
/// from then on, and the count of crashes, which stays until a request of the host.
pub fn start<B: Board>(board: &mut B) -> Boot {
    let (crash, failed) = decode(board.retained());
    board.set_retained(encode(Crash::None, failed));
    Boot { crash, failed }
}

/// A request of the host arrived: the firmware works well enough to be told to reset.
pub fn contact<B: Board>(board: &mut B) {
    let (crash, failed) = decode(board.retained());
    if failed != 0 {
        board.set_retained(encode(crash, 0));
    }
    board.contact();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_byte_without_the_mark_means_nothing() {
        assert_eq!(decode(0), (Crash::None, 0));
        assert_eq!(decode(0xFF), (Crash::None, 0));
        assert_eq!(decode(0xB0), (Crash::None, 0));
    }

    #[test]
    fn encode_and_decode_agree() {
        for crash in [Crash::None, Crash::Panic, Crash::Fault, Crash::Watchdog] {
            for failed in 0..=3 {
                assert_eq!(decode(encode(crash, failed)), (crash, failed));
            }
        }
        assert_eq!(decode(encode(Crash::Panic, 9)), (Crash::Panic, 3));
    }

    #[test]
    fn crashes_are_counted_where_they_happen() {
        // After a power-on, and from a byte another firmware left.
        for first in [0, 0xFF, 0xB0] {
            let verdict = crash(first, Crash::Panic);
            assert_eq!(decode(verdict.retained), (Crash::Panic, 1));
            assert_eq!(verdict.to, Reset::Application);
        }
        // Crash after crash, without the runner ever starting (so without `start`): the third
        // one goes to the bootloader, with a count that starts again.
        let mut retained = 0;
        let mut resets = [Reset::Application; 4];
        for reset in &mut resets {
            let verdict = crash(retained, Crash::Watchdog);
            retained = verdict.retained;
            *reset = verdict.to;
        }
        assert_eq!(
            resets,
            [
                Reset::Application,
                Reset::Application,
                Reset::Bootloader,
                Reset::Application
            ]
        );
        assert_eq!(decode(retained), (Crash::Watchdog, 1));
    }
}
