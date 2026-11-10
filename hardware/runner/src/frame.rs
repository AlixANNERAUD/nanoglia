//! The frames of the link, and the helpers that read and write their payloads.
//!
//! Every message, in both directions, is one frame (integers are little-endian):
//!
//! | Offset | Size | Field |
//! | --- | --- | --- |
//! | 0 | 2 | [`SOF`], `"NG"` |
//! | 2 | 1 | [`VERSION`] of the protocol |
//! | 3 | 1 | kind: a [`command`](crate::command) in a request, the command with [`RESPONSE`] set in its response, [`ERROR`] in an error |
//! | 4 | 1 | sequence number, chosen by the host and copied into the response |
//! | 5 | 2 | length `n` of the payload, at most [`MAX_PAYLOAD`] |
//! | 7 | `n` | payload |
//! | 7 + `n` | 4 | CRC-32 (the one of zlib) of bytes 2 to 7 + `n` |
//!
//! The host sends a request and waits for its response before the next one: a payload never
//! exceeds [`MAX_PAYLOAD`], so anything larger (a model, an output) goes in several requests, each
//! acknowledged by its response. Every request can be sent again: a write names its offset.

/// Start of a frame.
pub const SOF: [u8; 2] = *b"NG";
/// Version of the protocol. A frame of another version gets an [`ErrorCode::Version`] error.
pub const VERSION: u8 = 1;
/// Largest payload of a frame, in bytes.
pub const MAX_PAYLOAD: usize = 1024;
/// Bytes before the payload.
pub const HEADER: usize = 7;
/// Bytes of the CRC after the payload.
pub const TRAILER: usize = 4;
/// Largest frame.
pub const MAX_FRAME: usize = HEADER + MAX_PAYLOAD + TRAILER;
/// Bit of the kind set in the response to a command.
pub const RESPONSE: u8 = 0x80;
/// Kind of an error frame: a `u8` [`ErrorCode`], then a message in UTF-8.
pub const ERROR: u8 = 0xFF;

/// Why the device answers a request with an [`ERROR`] frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ErrorCode {
    /// The length of the frame exceeds [`MAX_PAYLOAD`].
    BadFrame = 1,
    /// The CRC of the frame is wrong.
    BadCrc = 2,
    /// The frame has another [`VERSION`].
    Version = 3,
    /// No such command.
    UnknownCommand = 4,
    /// The payload is malformed, or names memory outside the RAM store or a model the firmware
    /// does not have.
    BadArgument = 5,
    /// The rest of a frame did not arrive in time: the device dropped what it had.
    Timeout = 6,
    /// The command needs the results of a run, and none is at hand.
    NoRun = 7,
}

/// The state of a CRC-32 before any byte.
pub const CRC_INIT: u32 = 0xFFFF_FFFF;

/// Half-byte table of the reflected CRC-32 polynomial `0xEDB88320`.
const CRC_TABLE: [u32; 16] = [
    0x0000_0000,
    0x1DB7_1064,
    0x3B6E_20C8,
    0x26D9_30AC,
    0x76DC_4190,
    0x6B6B_51F4,
    0x4DB2_6158,
    0x5005_713C,
    0xEDB8_8320,
    0xF00F_9344,
    0xD6D6_A3E8,
    0xCB61_B38C,
    0x9B64_C2B0,
    0x86D3_D2D4,
    0xA00A_E278,
    0xBDBD_F21C,
];

/// Feeds `bytes` to a CRC-32 whose state is `state` (start from [`CRC_INIT`]); [`crc_finish`]
/// gives the checksum.
pub fn crc_update(mut state: u32, bytes: &[u8]) -> u32 {
    for &byte in bytes {
        state ^= u32::from(byte);
        state = (state >> 4) ^ CRC_TABLE[(state & 0xF) as usize];
        state = (state >> 4) ^ CRC_TABLE[(state & 0xF) as usize];
    }
    state
}

/// The checksum of a CRC-32 state.
pub const fn crc_finish(state: u32) -> u32 {
    !state
}

/// The CRC-32 of `bytes`, as zlib computes it.
pub fn crc32(bytes: &[u8]) -> u32 {
    crc_finish(crc_update(CRC_INIT, bytes))
}

/// Writes the frame of `kind` and `seq` around the `len` bytes of payload already at
/// `frame[HEADER..]`, and returns the length of the frame. `None` if the payload is too long or
/// the buffer too short.
pub fn seal(frame: &mut [u8], kind: u8, seq: u8, len: usize) -> Option<usize> {
    if len > MAX_PAYLOAD {
        return None;
    }
    let end = HEADER + len;
    let header = frame.get_mut(..HEADER)?;
    header[..2].copy_from_slice(&SOF);
    header[2] = VERSION;
    header[3] = kind;
    header[4] = seq;
    header[5..7].copy_from_slice(&(len as u16).to_le_bytes());
    let crc = crc32(frame.get(2..end)?);
    frame
        .get_mut(end..end + TRAILER)?
        .copy_from_slice(&crc.to_le_bytes());
    Some(end + TRAILER)
}

/// Reads the fields of a payload in order. Every method returns `None` past the end.
pub struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    /// A reader at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// The next `len` bytes.
    pub fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let (head, tail) = self.bytes.split_at_checked(len)?;
        self.bytes = tail;
        Some(head)
    }

    /// The next byte.
    pub fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }

    /// The next little-endian `u16`.
    pub fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    /// The next little-endian `u32`.
    pub fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    /// The next little-endian `u32`, as a `usize`.
    pub fn size(&mut self) -> Option<usize> {
        self.u32().map(|v| v as usize)
    }

    /// Everything left.
    pub fn rest(&mut self) -> &'a [u8] {
        core::mem::take(&mut self.bytes)
    }

    /// Whether nothing is left.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Writes the fields of a payload in order. What does not fit is dropped and remembered:
/// [`Writer::finish`] then returns `None`.
pub struct Writer<'a> {
    bytes: &'a mut [u8],
    len: usize,
    overflow: bool,
}

impl<'a> Writer<'a> {
    /// A writer at the start of `bytes`.
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes,
            len: 0,
            overflow: false,
        }
    }

    /// Bytes left.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.len
    }

    /// Appends `bytes`, all or nothing.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        match self.bytes.get_mut(self.len..self.len + bytes.len()) {
            Some(slot) => {
                slot.copy_from_slice(bytes);
                self.len += bytes.len();
            }
            None => self.overflow = true,
        }
        self
    }

    /// Appends a byte.
    pub fn u8(&mut self, value: u8) -> &mut Self {
        self.bytes(&[value])
    }

    /// Appends a little-endian `u32`.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.bytes(&value.to_le_bytes())
    }

    /// Appends a `usize` as a little-endian `u32`, saturated.
    pub fn size(&mut self, value: usize) -> &mut Self {
        self.u32(u32::try_from(value).unwrap_or(u32::MAX))
    }

    /// Appends a string of at most 255 bytes (cut on a character boundary), after its length in a
    /// byte.
    pub fn string(&mut self, text: &str) -> &mut Self {
        let mut end = text.len().min(255);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        if self.remaining() < 1 + end {
            self.overflow = true;
            return self;
        }
        self.u8(end as u8).bytes(&text.as_bytes()[..end])
    }

    /// As much of `text` as fits, cut on a character boundary, without a length: for the message
    /// that ends a payload.
    pub fn text(&mut self, text: &str) -> &mut Self {
        let mut end = text.len().min(self.remaining());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.bytes(&text.as_bytes()[..end])
    }

    /// The length written, or `None` if something did not fit.
    pub fn finish(&self) -> Option<usize> {
        (!self.overflow).then_some(self.len)
    }
}

/// Formats into a [`Writer`], truncating: for error messages.
impl core::fmt::Write for Writer<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.text(s);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        // The check value of CRC-32/ISO-HDLC, and zlib.crc32 of two other inputs.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(&[0u8; 32]), 0x190A_55AD);
        // Incremental and one-shot agree.
        let state = crc_update(crc_update(CRC_INIT, b"1234"), b"56789");
        assert_eq!(crc_finish(state), 0xCBF4_3926);
    }

    #[test]
    fn seal_writes_header_and_crc() {
        let mut frame = [0u8; MAX_FRAME];
        frame[HEADER..HEADER + 3].copy_from_slice(b"abc");
        let len = seal(&mut frame, 0x81, 7, 3).unwrap();
        assert_eq!(len, HEADER + 3 + TRAILER);
        assert_eq!(&frame[..7], &[b'N', b'G', VERSION, 0x81, 7, 3, 0]);
        let crc = u32::from_le_bytes(frame[10..14].try_into().unwrap());
        assert_eq!(crc, crc32(&frame[2..10]));
        assert_eq!(seal(&mut frame, 0, 0, MAX_PAYLOAD + 1), None);
        assert_eq!(seal(&mut frame[..8], 0, 0, 3), None);
    }

    #[test]
    fn reader_stops_at_the_end() {
        let mut reader = Reader::new(&[1, 2, 0, 3, 0, 0, 0, 9]);
        assert_eq!(reader.u8(), Some(1));
        assert_eq!(reader.u16(), Some(2));
        assert_eq!(reader.u32(), Some(3));
        assert_eq!(reader.u32(), None);
        assert_eq!(reader.rest(), &[9]);
        assert!(reader.is_empty());
        assert_eq!(reader.u8(), None);
    }

    #[test]
    fn writer_reports_overflow() {
        let mut buffer = [0u8; 8];
        let mut writer = Writer::new(&mut buffer);
        writer.u32(1).string("ab");
        assert_eq!(writer.finish(), Some(7));
        writer.u32(2);
        assert_eq!(writer.finish(), None);

        // A message is cut on a character boundary, without overflow.
        let mut writer = Writer::new(&mut buffer[..3]);
        writer.text("aéé");
        assert_eq!(writer.finish(), Some(3));
        assert_eq!(&buffer[..3], "aé".as_bytes());
    }
}
