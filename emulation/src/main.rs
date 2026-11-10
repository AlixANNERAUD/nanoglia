//! Runs one golden case of `testdata/` under an instruction-set emulator.
//!
//! Before the reset, `tools/run.py` writes the case at `_case_start` (a region of the linker
//! script): a header of little-endian `u32`s, then the model and the inputs.
//!
//! | Offset | Field |
//! | --- | --- |
//! | 0 | [`MAGIC`] |
//! | 4, 8 | model offset (from `_case_start`, 16-byte aligned) and length |
//! | 12 | input count `n` |
//! | 16 + 8 i | offset and length of input `i` |
//!
//! The firmware loads the model with every kernel of nanoglia (CMSIS-NN with the DSP or the
//! portable code paths the target selects, the Rust kernels, float32, int16), plans its memory in
//! the arena between `_arena_start` and `_arena_end`, runs it, then fills [`EMULATION_RESULT`]
//! (the address and length of each output, left in the arena, and the kernel of each operator)
//! and writes [`EMULATION_STATUS`], the write that ends the emulation. Nothing is compared here:
//! the host compares the outputs.
//!
//! The emulator checks the instructions, not the core around them: Unicorn does not model timing,
//! caches or flash wait states, and it takes alignment faults on its Cortex-M0 only: it runs an
//! unaligned `ldm` on the Cortex-M4 or M33, or a misaligned load on RISC-V, without a fault.

#![no_std]
#![no_main]

use core::fmt::Write;
use core::mem::MaybeUninit;
use core::ptr::addr_of_mut;
use core::sync::atomic::{Ordering, compiler_fence};
use nanoglia::{Arena, Encoding, Graph, Target};

#[cfg(target_arch = "arm")]
use cortex_m_rt::entry;
#[cfg(target_arch = "riscv32")]
use riscv_rt::entry;

/// `"NGCS"`, the first word of a case.
const MAGIC: u32 = u32::from_le_bytes(*b"NGCS");
/// Outputs a case may have.
const MAX_OUTPUTS: usize = 16;
/// Bit of [`Report::truncated`]: the kernel names did not fit in [`Report::kernels`].
const TRUNCATED_KERNELS: u32 = 1;
/// Bit of [`Report::truncated`]: the message did not fit in [`Report::message`].
const TRUNCATED_MESSAGE: u32 = 2;

/// How the run ended, read by the host.
#[repr(u32)]
#[derive(Clone, Copy)]
enum Status {
    /// Every output is in [`EMULATION_RESULT`].
    Done = 1,
    /// nanoglia returned an error, described in the message.
    Error = 2,
    /// The firmware panicked, described in the message.
    Panic = 3,
    /// The processor took a fault, at the address in the message.
    Fault = 4,
}

/// What the host reads after the run, besides the status.
#[repr(C)]
pub struct Report {
    /// Number of outputs.
    output_count: u32,
    /// Address and length in bytes of each output.
    outputs: [[u32; 2]; MAX_OUTPUTS],
    /// Bytes of the arena the graph and its context took.
    arena_used: u32,
    /// Length of `kernels` in bytes.
    kernels_len: u32,
    /// The kernel of each operator ([`Graph::kernel_name`]), one per line, in UTF-8.
    kernels: [u8; 4096],
    /// Length of `message` in bytes.
    message_len: u32,
    /// The error or panic, in UTF-8.
    message: [u8; 256],
    /// [`TRUNCATED_KERNELS`] and [`TRUNCATED_MESSAGE`]: which of the two did not fit.
    truncated: u32,
}

/// The result of the case.
#[unsafe(no_mangle)]
pub static mut EMULATION_RESULT: Report = Report {
    output_count: 0,
    outputs: [[0; 2]; MAX_OUTPUTS],
    arena_used: 0,
    kernels_len: 0,
    kernels: [0; 4096],
    message_len: 0,
    message: [0; 256],
    truncated: 0,
};

/// 0 while running, then a [`Status`]. The start-up code zeroes it, so only a non-zero write ends
/// the emulation.
#[unsafe(no_mangle)]
pub static mut EMULATION_STATUS: u32 = 0;

unsafe extern "C" {
    static mut _arena_start: MaybeUninit<u8>;
    static _arena_end: u8;
    static _case_start: u8;
    static _case_end: u8;
}

/// Appends `s` to `buffer`, whose length is `len`, truncated to its size. Returns whether all of
/// `s` fit.
fn append(buffer: &mut [u8], len: &mut u32, s: &str) -> bool {
    for &byte in s.as_bytes() {
        let Some(slot) = buffer.get_mut(*len as usize) else {
            return false;
        };
        *slot = byte;
        *len += 1;
    }
    true
}

/// Writes text into the message of [`EMULATION_RESULT`], truncated to its size.
struct Message;

impl Write for Message {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        // SAFETY: single-threaded firmware, nothing else holds a reference to the report.
        let report = unsafe { &mut *addr_of_mut!(EMULATION_RESULT) };
        if !append(&mut report.message, &mut report.message_len, s) {
            report.truncated |= TRUNCATED_MESSAGE;
        }
        Ok(())
    }
}

/// Publishes the status and stops.
fn finish(status: Status) -> ! {
    // The host stops the emulation on this write and then reads the report, so every store to the
    // report must come before it. A volatile write is only ordered with other volatile accesses,
    // and the report is written with plain stores: the fence keeps the compiler from moving them
    // after it. (A single core with no cache between it and the host: no hardware barrier needed.)
    compiler_fence(Ordering::SeqCst);
    // SAFETY: single-threaded firmware, nothing else accesses the status.
    unsafe { addr_of_mut!(EMULATION_STATUS).write_volatile(status as u32) };
    loop {
        core::hint::spin_loop();
    }
}

/// Reports an error of `stage` and stops.
fn fail(stage: core::fmt::Arguments<'_>, error: impl core::fmt::Display) -> ! {
    let _ = write!(Message, "{stage}: {error}");
    finish(Status::Error)
}

/// The little-endian `u32` at `offset` of the case.
fn word(case: &[u8], offset: usize) -> Option<usize> {
    let bytes = case.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?) as usize)
}

/// The bytes at the offset and length stored at `field` of the case header.
fn span(case: &[u8], field: usize) -> Option<&[u8]> {
    let (offset, len) = (word(case, field)?, word(case, field + 4)?);
    case.get(offset..offset.checked_add(len)?)
}

#[entry]
fn main() -> ! {
    // SAFETY: the emulator maps both regions before the reset; nothing else uses them.
    let (case, memory) = unsafe {
        let case_start = &raw const _case_start;
        let case_len = (&raw const _case_end).addr() - case_start.addr();
        let arena_start = &raw mut _arena_start;
        let arena_len = (&raw const _arena_end).addr() - arena_start.addr();
        (
            core::slice::from_raw_parts(case_start, case_len),
            core::slice::from_raw_parts_mut(arena_start, arena_len),
        )
    };
    if word(case, 0) != Some(MAGIC as usize) {
        fail(format_args!("case"), "no case header");
    }
    let Some(model) = span(case, 4) else {
        fail(format_args!("case"), "model out of the case region")
    };

    let arena = Arena::new_uninit(memory);
    let graph = Graph::load_in(&[model], Encoding::Autodetect, Target::Auto, &arena)
        .unwrap_or_else(|e| fail(format_args!("load"), e));
    let mut ctx = graph
        .init_execution_context_in(&arena)
        .unwrap_or_else(|e| fail(format_args!("context"), e));

    let inputs = word(case, 12).unwrap_or(0);
    for i in 0..inputs {
        let Some(data) = span(case, 16 + 8 * i) else {
            fail(format_args!("input {i}"), "out of the case region")
        };
        ctx.set_input(i, data)
            .unwrap_or_else(|e| fail(format_args!("input {i}"), e));
    }
    ctx.compute()
        .unwrap_or_else(|e| fail(format_args!("compute"), e));

    // SAFETY: single-threaded firmware, nothing else holds a reference to the report.
    let report = unsafe { &mut *addr_of_mut!(EMULATION_RESULT) };
    let outputs = graph.output_count();
    if outputs > MAX_OUTPUTS {
        fail(format_args!("outputs"), "more than MAX_OUTPUTS");
    }
    for (i, slot) in report.outputs.iter_mut().enumerate().take(outputs) {
        let output = ctx
            .output(i)
            .unwrap_or_else(|e| fail(format_args!("output {i}"), e));
        let bytes = output.bytes();
        *slot = [bytes.as_ptr().addr() as u32, bytes.len() as u32];
    }
    report.output_count = outputs as u32;
    report.arena_used = arena.used() as u32;
    for i in 0..graph.operator_count() {
        let name = graph.kernel_name(i).unwrap_or("?");
        let line = append(&mut report.kernels, &mut report.kernels_len, name)
            && append(&mut report.kernels, &mut report.kernels_len, "\n");
        if !line {
            report.truncated |= TRUNCATED_KERNELS;
            break;
        }
    }
    finish(Status::Done)
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    let _ = write!(Message, "panic: {info}");
    finish(Status::Panic)
}

/// A fault (undefined instruction, bad address...): reported with the address of the faulting
/// instruction rather than left spinning until the timeout. Unicorn stops the emulation itself on
/// most faults, and takes no alignment fault on the Cortex-M4 and M33 (see the crate
/// documentation).
#[cfg(target_arch = "arm")]
#[cortex_m_rt::exception]
unsafe fn HardFault(frame: &cortex_m_rt::ExceptionFrame) -> ! {
    let _ = write!(Message, "hard fault at pc=0x{:08x}", frame.pc());
    finish(Status::Fault)
}

/// An exception (illegal instruction, bad address...): reported with the address of the faulting
/// instruction rather than left spinning until the timeout. Unicorn runs a misaligned access
/// instead of raising the exception a core without misaligned-access support would (see the crate
/// documentation).
#[cfg(target_arch = "riscv32")]
#[unsafe(export_name = "ExceptionHandler")]
fn exception_handler(_frame: &riscv_rt::TrapFrame) -> ! {
    let _ = write!(
        Message,
        "exception {} at pc=0x{:08x}",
        riscv::register::mcause::read().code(),
        riscv::register::mepc::read()
    );
    finish(Status::Fault)
}
