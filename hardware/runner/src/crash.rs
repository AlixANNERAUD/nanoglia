//! The message of a crash, left in RAM for the next run to report.
//!
//! A panic or a fault cannot count on the link: the handler writes a record in a part of the RAM
//! the start-up code does not clear, notes the crash in the retained byte ([`crate::boot`]) and
//! resets; the firmware reports the record in its next answer to `INFO`. The record is checked by
//! a magic word and a CRC: a bootloader that ran in between may have used this memory, in which
//! case the message is lost and only the kind of the crash, in the retained byte, remains.
//!
//! The linker script of a firmware places the section `.nanoglia_crash` where its bootloader does
//! not write: the Arduino bootloader of the Nano 33 BLE clears and uses the RAM from `0x20000000`
//! to `0x20009e14` on every reset (the `bss`, `noinit` and data sections of its published ELF),
//! so the record lives in the last 256 bytes of the RAM, above the stack.

use crate::frame::crc32;
use core::fmt::{self, Write};
use core::mem::MaybeUninit;
use core::ptr::addr_of_mut;

/// `"NGCR"`.
const MAGIC: u32 = u32::from_le_bytes(*b"NGCR");
/// Longest message kept.
pub const MAX_MESSAGE: usize = 160;

#[repr(C)]
struct Record {
    magic: u32,
    len: u32,
    crc: u32,
    message: [u8; MAX_MESSAGE],
}

/// The record. On a bare-metal target, in a section of its own, which the start-up code neither
/// loads nor clears.
#[cfg_attr(target_os = "none", unsafe(link_section = ".nanoglia_crash"))]
static mut RECORD: MaybeUninit<Record> = MaybeUninit::zeroed();

struct Message<'a> {
    bytes: &'a mut [u8; MAX_MESSAGE],
    len: usize,
}

impl Write for Message<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let mut end = s.len().min(MAX_MESSAGE - self.len);
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        self.bytes[self.len..self.len + end].copy_from_slice(&s.as_bytes()[..end]);
        self.len += end;
        Ok(())
    }
}

/// Keeps `message`, truncated to [`MAX_MESSAGE`] bytes, for [`take`] after the reset.
///
/// For the panic and fault handlers, which run alone: nothing else may be using the record.
pub fn store(message: fmt::Arguments<'_>) {
    let mut bytes = [0u8; MAX_MESSAGE];
    let mut writer = Message {
        bytes: &mut bytes,
        len: 0,
    };
    let _ = writer.write_fmt(message);
    let len = writer.len;
    let record = Record {
        magic: MAGIC,
        len: len as u32,
        crc: crc32(&bytes[..len]),
        message: bytes,
    };
    // SAFETY: the record is plain bytes; the firmware is single-threaded and the caller is a
    // handler that never returns, so no reference to it is alive.
    unsafe { addr_of_mut!(RECORD).write_volatile(MaybeUninit::new(record)) };
}

/// Copies the message of the previous run into `out` and forgets it. Returns its length, 0 if
/// there is none (or if it does not check).
///
/// Call it once, when the firmware starts, before anything can [`store`].
pub fn take(out: &mut [u8; MAX_MESSAGE]) -> usize {
    // SAFETY: every bit pattern is a valid `Record` (integers and bytes); after a reset the
    // memory holds whatever the previous run or a bootloader left, which the magic word and the
    // CRC sort out. Single-threaded, no other reference.
    let record = unsafe { addr_of_mut!(RECORD).read_volatile().assume_init() };
    // SAFETY: as above; the magic word is cleared so that the message is reported once.
    unsafe { addr_of_mut!(RECORD).write_volatile(MaybeUninit::zeroed()) };
    let len = record.len as usize;
    if record.magic != MAGIC || len > MAX_MESSAGE || crc32(&record.message[..len]) != record.crc {
        return 0;
    }
    // A message cut by `store` ends on a character boundary; anything else is not a record.
    if core::str::from_utf8(&record.message[..len]).is_err() {
        return 0;
    }
    out[..len].copy_from_slice(&record.message[..len]);
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test: the record is a single static.
    #[test]
    fn a_stored_message_is_taken_once() {
        let mut out = [0u8; MAX_MESSAGE];
        assert_eq!(take(&mut out), 0);

        store(format_args!("panic at {}:{}", "kernel.rs", 12));
        let len = take(&mut out);
        assert_eq!(&out[..len], b"panic at kernel.rs:12");
        assert_eq!(take(&mut out), 0);

        // Truncated on a character boundary.
        let long = "é".repeat(MAX_MESSAGE);
        store(format_args!("{long}"));
        let len = take(&mut out);
        assert_eq!(len, MAX_MESSAGE);
        assert!(core::str::from_utf8(&out[..len]).is_ok());

        // A record whose CRC does not check is no record.
        store(format_args!("message"));
        // SAFETY: test only, single-threaded access to the record.
        unsafe { (*addr_of_mut!(RECORD)).assume_init_mut().message[0] ^= 1 };
        assert_eq!(take(&mut out), 0);
    }
}
