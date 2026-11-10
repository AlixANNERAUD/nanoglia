//! The byte that survives a reset: how the previous run ended, and how many runs in a row ended in
//! a crash before the host said a word.
//!
//! A firmware that crashes while it starts would reset forever, out of reach of the host. So each
//! crash is counted, the count is cleared when a request of the host arrives, and at
//! [`MAX_FAILED_BOOTS`] the firmware resets into the bootloader instead, where a new firmware can
//! be flashed.
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

/// The retained byte a crash handler writes before it resets: `retained` with `crash` noted.
pub fn note_crash(retained: u8, crash: Crash) -> u8 {
    encode(crash, decode(retained).1)
}

/// What the retained byte says when the firmware starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boot {
    /// How the previous run ended.
    pub crash: Crash,
    /// Crashes in a row without a request of the host, this one included.
    pub failed: u8,
}

/// Reads the retained byte when the firmware starts, counts the crash of the previous run if there
/// was one, and resets into the bootloader at [`MAX_FAILED_BOOTS`].
pub fn start<B: Board>(board: &mut B) -> Boot {
    let (crash, mut failed) = decode(board.retained());
    if crash != Crash::None {
        failed += 1;
    }
    if failed >= MAX_FAILED_BOOTS {
        // The count starts again: the bootloader runs this firmware if none replaces it.
        board.set_retained(encode(Crash::None, 0));
        board.reset(Reset::Bootloader);
        // Only the tests come here.
        return Boot { crash, failed };
    }
    board.set_retained(encode(Crash::None, failed));
    Boot { crash, failed }
}

/// A request of the host arrived: the firmware works well enough to be told to reset.
pub fn contact<B: Board>(board: &mut B) {
    let (crash, failed) = decode(board.retained());
    if failed != 0 {
        board.set_retained(encode(crash, 0));
    }
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
    fn a_crash_keeps_the_count() {
        assert_eq!(decode(note_crash(0, Crash::Panic)), (Crash::Panic, 0));
        let two = encode(Crash::None, 2);
        assert_eq!(
            decode(note_crash(two, Crash::Watchdog)),
            (Crash::Watchdog, 2)
        );
    }
}
