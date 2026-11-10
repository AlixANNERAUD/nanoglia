//! The runner of the golden cases on an emulated Cortex-M4F: the code of the board firmware,
//! minus the board. The link, the clock and the resets are registers the emulator
//! (`tools/nanoglia_hardware/emulated.py`, on the Unicorn engine) answers:
//!
//! | Address | Register |
//! | --- | --- |
//! | `0x4000_0000` | read: bytes of the host waiting (0 or 1) |
//! | `0x4000_0004` | read: the next byte of the host |
//! | `0x4000_0008` | write: a byte for the host |
//! | `0x4000_000C` | read: the cycle counter, which the emulator moves at each reading |
//! | `0x4000_0010` | write: reset, 1 into the firmware, 2 into the bootloader |
//! | `0x4000_0014` | the byte that survives a reset |
//! | `0x4000_0018` | read: the time in milliseconds, which the emulator moves when the host waits |
//! | `0x4000_001C` | read: 1 to panic while starting, before the runner exists, for the test of the boot loop |
//!
//! When it has nothing to read, the firmware calls [`emulated_wait`]: the emulator stops there
//! and gives the hand back to the host.
//!
//! The emulator checks the instructions, not the core around them: no timing, no interrupt (so no
//! watchdog: a hang the host asks for ends as the watchdog of a board would end it), no USB.

#![no_std]
#![no_main]

use core::fmt;
use cortex_m_rt::entry;
use nanoglia_hardware_runner::boot::{self, Crash};
use nanoglia_hardware_runner::{
    Board, BoardInfo, EmbeddedModel, Fault, LinkError, Reset, Runner, crash, include_model, stack,
};

const REGISTERS: usize = 0x4000_0000;
const RX_READY: usize = REGISTERS;
const RX_DATA: usize = REGISTERS + 0x04;
const TX_DATA: usize = REGISTERS + 0x08;
const CYCLES: usize = REGISTERS + 0x0C;
const RESET: usize = REGISTERS + 0x10;
const RETAINED: usize = REGISTERS + 0x14;
const TIME_MS: usize = REGISTERS + 0x18;
const INJECT: usize = REGISTERS + 0x1C;

fn read(register: usize) -> u32 {
    // SAFETY: a register of the emulator, mapped before the reset.
    unsafe { (register as *const u32).read_volatile() }
}

fn write(register: usize, value: u32) {
    // SAFETY: a register of the emulator, mapped before the reset.
    unsafe { (register as *mut u32).write_volatile(value) }
}

/// Where the firmware waits for the host: the emulator stops on the first instruction.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn emulated_wait() {
    // Something the optimizer cannot remove nor merge with another function.
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    let _ = read(RX_READY);
}

/// The MLPerf Tiny models, linked in: they run in place from flash, as in an application.
static MODELS: [EmbeddedModel; 5] = [
    EmbeddedModel {
        name: "kws_ref_model.tflite",
        bytes: include_model!("../../../testdata/mlperf-tiny/kws_ref_model.tflite"),
    },
    EmbeddedModel {
        name: "pretrainedResnet_quant.tflite",
        bytes: include_model!("../../../testdata/mlperf-tiny/pretrainedResnet_quant.tflite"),
    },
    EmbeddedModel {
        name: "str_ww_ref_model.tflite",
        bytes: include_model!("../../../testdata/mlperf-tiny/str_ww_ref_model.tflite"),
    },
    EmbeddedModel {
        name: "ad01_int8.tflite",
        bytes: include_model!("../../../testdata/mlperf-tiny/ad01_int8.tflite"),
    },
    EmbeddedModel {
        name: "vww_96_int8.tflite",
        bytes: include_model!("../../../testdata/mlperf-tiny/vww_96_int8.tflite"),
    },
];

/// Counts the crash, leaves its message for the next run and resets, into the bootloader after
/// too many: as the handlers of a board do.
fn die(kind: Crash, message: fmt::Arguments<'_>) -> ! {
    crash::store(message);
    let verdict = boot::crash(read(RETAINED) as u8, kind);
    write(RETAINED, u32::from(verdict.retained));
    write(RESET, verdict.to as u32 + 1);
    // The emulator restarts the firmware on this write.
    loop {
        emulated_wait();
    }
}

struct Emulated;

impl Board for Emulated {
    fn info(&self) -> BoardInfo {
        BoardInfo {
            board: "emulated Cortex-M4F (Unicorn)",
            firmware: concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION")),
            clock: "a counter of the emulator, not a time",
            counter_hz: 0,
            reset_reason: 0,
            settings: "none: an emulator has no timing",
        }
    }

    fn read(&mut self, buffer: &mut [u8], timeout_ms: Option<u32>) -> Result<usize, LinkError> {
        let start = read(TIME_MS);
        while read(RX_READY) == 0 {
            if timeout_ms.is_some_and(|timeout| read(TIME_MS).wrapping_sub(start) >= timeout) {
                return Err(LinkError::Timeout);
            }
            emulated_wait();
        }
        let mut count = 0;
        for slot in buffer {
            if read(RX_READY) == 0 {
                break;
            }
            *slot = read(RX_DATA) as u8;
            count += 1;
        }
        Ok(count)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), LinkError> {
        for &byte in bytes {
            write(TX_DATA, u32::from(byte));
        }
        Ok(())
    }

    fn cycles(&mut self) -> u32 {
        read(CYCLES)
    }

    fn idle(&mut self) {}

    fn reset(&mut self, to: Reset) {
        write(RESET, to as u32 + 1);
        loop {
            emulated_wait();
        }
    }

    fn fault(&mut self, fault: Fault) {
        match fault {
            Fault::Panic => panic!("fault requested by the host"),
            // Unicorn does not take the exception of an undefined instruction: end as the fault
            // handler of a board does.
            Fault::HardFault => die(
                Crash::Fault,
                format_args!("hard fault requested by the host"),
            ),
            // No interrupt, so no watchdog: end as the watchdog of a board does.
            Fault::Hang => die(
                Crash::Watchdog,
                format_args!("watchdog: hang requested by the host"),
            ),
        }
    }

    fn retained(&mut self) -> u8 {
        read(RETAINED) as u8
    }

    fn set_retained(&mut self, value: u8) {
        write(RETAINED, u32::from(value));
    }

    fn stack_paint(&mut self) {
        stack::paint();
    }

    fn stack_peak(&mut self) -> Option<u32> {
        Some(stack::peak())
    }
}

unsafe extern "C" {
    static mut __sheap: u8;
    static _runner_stack_limit: u8;
}

#[entry]
fn main() -> ! {
    // The RAM store: from the end of the static data, on a 16-byte boundary, to the stack.
    let start = (&raw mut __sheap).addr().next_multiple_of(16);
    let len = (&raw const _runner_stack_limit).addr() - start;
    // SAFETY: the linker script leaves this memory to no one else, and `main` runs once.
    let ram = unsafe {
        let start = (&raw mut __sheap).with_addr(start);
        start.write_bytes(0, len);
        core::slice::from_raw_parts_mut(start, len)
    };
    // Before the runner exists, as a board that fails to start its clock or its port would.
    if read(INJECT) == 1 {
        panic!("panic while starting, injected by the emulator");
    }
    let mut runner = Runner::new(Emulated, ram, &MODELS);
    runner.serve();
    loop {
        emulated_wait();
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    die(Crash::Panic, format_args!("{info}"))
}

/// A fault, reported like a panic. Unicorn stops the emulation itself on most faults.
#[cortex_m_rt::exception]
unsafe fn HardFault(frame: &cortex_m_rt::ExceptionFrame) -> ! {
    die(
        Crash::Fault,
        format_args!("hard fault at pc=0x{:08x}", frame.pc()),
    )
}
