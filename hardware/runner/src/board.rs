//! What the runner needs from a board: the only code that differs between the boards, the
//! emulated one and the tests.

/// Why a read or a write of the link ended early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkError {
    /// Nothing moved in time.
    Timeout,
    /// The link is gone for good. A board never returns it; the tests do, to end the runner.
    Closed,
}

/// Where a reset leads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reset {
    /// The firmware starts again.
    Application = 0,
    /// The bootloader of the board waits for a new firmware.
    Bootloader = 1,
}

/// A failure the host asks for, to check that the board comes back from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Fault {
    /// A Rust panic.
    Panic = 1,
    /// A processor fault (an undefined instruction).
    HardFault = 2,
    /// A loop that never ends: the watchdog must end it.
    Hang = 3,
}

/// What a board says about itself in the answer to `INFO`.
#[derive(Clone, Copy, Debug)]
pub struct BoardInfo {
    /// Name of the board.
    pub board: &'static str,
    /// Name and version of the firmware.
    pub firmware: &'static str,
    /// The clock [`Board::cycles`] counts, in words.
    pub clock: &'static str,
    /// Ticks of [`Board::cycles`] per second.
    pub counter_hz: u32,
    /// Why the board last reset, in the encoding of the chip (`RESETREAS` on an nRF52).
    pub reset_reason: u32,
    /// What else decides the times measured on this board, in words (the state of a cache...).
    pub settings: &'static str,
}

/// A board: its link to the host, its cycle counter and its ways out.
///
/// The link is a byte stream without framing nor flow control (a USB CDC-ACM port, a UART).
pub trait Board {
    /// What the board says about itself.
    fn info(&self) -> BoardInfo;

    /// Reads at least one byte and at most `buffer.len()` into `buffer`, waiting `timeout_ms`
    /// milliseconds at most for the first one (forever with `None`), and returns their number.
    ///
    /// Waiting keeps the board alive: it does what [`Board::idle`] does.
    fn read(&mut self, buffer: &mut [u8], timeout_ms: Option<u32>) -> Result<usize, LinkError>;

    /// Writes all of `bytes`, one whole frame. A host that does not read them in time is a
    /// [`LinkError::Timeout`]: the frame is lost, the runner goes on.
    fn write(&mut self, bytes: &[u8]) -> Result<(), LinkError>;

    /// A free-running counter of the cycles of the core clock, which wraps around.
    fn cycles(&mut self) -> u32;

    /// Called before each operator of a run: the run is making progress. A board feeds its watchdog and serves its link here.
    fn idle(&mut self);

    /// Resets the board. Does not return on a board; the runner stops if it does.
    fn reset(&mut self, to: Reset);

    /// Takes the fault `fault`. Does not return on a board.
    fn fault(&mut self, fault: Fault) {
        match fault {
            Fault::Panic | Fault::HardFault => panic!("fault requested by the host"),
            Fault::Hang => loop {
                core::hint::spin_loop();
            },
        }
    }

    /// A request of the host arrived, for the first time since the firmware started: the link
    /// works.
    fn contact(&mut self) {}

    /// The byte that survives a reset, see [`crate::boot`]. 0 after a power-on.
    fn retained(&mut self) -> u8;

    /// Sets the byte that survives a reset.
    fn set_retained(&mut self, value: u8);

    /// Fills the unused part of the stack with a pattern, for [`Board::stack_peak`].
    fn stack_paint(&mut self) {}

    /// The deepest the stack went since [`Board::stack_paint`], in bytes, if the board measures
    /// it.
    fn stack_peak(&mut self) -> Option<u32> {
        None
    }
}

/// A model linked into the firmware: it stays in flash and runs in place, as in an application.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddedModel {
    /// The name of its file, without the directory.
    pub name: &'static str,
    /// Its bytes, aligned on 16 bytes.
    pub bytes: &'static [u8],
}

/// The bytes of a model file, aligned on 16 bytes as the constants of a model must be, for
/// [`EmbeddedModel::bytes`].
#[macro_export]
macro_rules! include_model {
    ($path:expr) => {{
        #[repr(C, align(16))]
        struct Aligned<T: ?Sized>(T);
        static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!($path));
        &MODEL.0
    }};
}
