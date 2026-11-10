//! How deep the stack of a Cortex-M firmware went, by painting it: for [`Board::stack_paint`] and
//! [`Board::stack_peak`](crate::Board::stack_peak).
//!
//! The stack lies between `_runner_stack_limit`, a symbol of the linker script of the firmware,
//! and `_stack_start`, where cortex-m-rt starts it; it grows down. Nothing checks that it stays
//! above the limit: what lies below is the RAM store, whose end the arena of a model seldom
//! reaches, and a peak close to the size of the stack is the sign to look for.
//!
//! [`Board::stack_paint`]: crate::Board::stack_paint

use core::arch::asm;

/// What the unused stack is filled with.
const PATTERN: u32 = 0xA5C3_5AE1;
/// Bytes left alone under the stack pointer: the frames of the painter itself.
const MARGIN: usize = 64;

unsafe extern "C" {
    static mut _runner_stack_limit: u32;
    static _stack_start: u32;
}

fn limit() -> *mut u32 {
    (&raw mut _runner_stack_limit).cast()
}

/// Fills the stack below the stack pointer with the pattern.
#[inline(never)]
pub fn paint() {
    let sp: usize;
    // SAFETY: reads the main stack pointer, nothing else.
    unsafe { asm!("mrs {}, MSP", out(reg) sp, options(nomem, nostack, preserves_flags)) };
    let end = (sp.saturating_sub(MARGIN) & !3) as *mut u32;
    let mut word = limit();
    while word < end {
        // SAFETY: between the limit of the stack and the stack pointer, memory nothing uses: the
        // frames of this function are above `end`, and an interrupt taken meanwhile has returned
        // by the time its frame is painted over.
        unsafe {
            word.write_volatile(PATTERN);
            word = word.add(1);
        }
    }
}

/// The deepest the stack went since [`paint`], in bytes: the size of the stack if it reached its
/// limit, or went past it.
pub fn peak() -> u32 {
    let top = (&raw const _stack_start) as usize;
    let mut word = limit();
    // SAFETY: reads the stack region, word by word, up to its top.
    while (word as usize) < top && unsafe { word.read_volatile() } == PATTERN {
        // SAFETY: stays within the stack region, see above.
        word = unsafe { word.add(1) };
    }
    (top - word as usize) as u32
}
