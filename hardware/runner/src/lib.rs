//! Runs golden cases of nanoglia on a board, on the orders of a host at the other end of a byte
//! stream: the protocol and the runner, shared by the board firmwares, the emulated one and the
//! tests. A board only brings a [`Board`]: its link, its cycle counter and its ways out.
//!
//! The host names a model and writes its inputs into the RAM of the board, asks for a run, then
//! reads the outputs, the kernel and the cycles of each operator. Nothing is compared here: the
//! host compares the outputs.
//!
//! # Where the model is
//!
//! | Store | What |
//! | --- | --- |
//! | [`store::RAM`] | A block of RAM. The host lays it out: a model and its inputs first, then, from the arena offset of `RUN` to its end, the arena of the graph and its execution context. A model received here checks the results only: an application keeps its model in flash, and a kernel that reads its weights from RAM is not timed as it would be there. |
//! | [`store::EMBEDDED`] | The models linked into the firmware ([`EmbeddedModel`]), which run in place from flash, as in an application: the only ones whose times mean something. `INFO` lists them. |
//!
//! # Commands
//!
//! Frames are described in [`frame`]; integers are little-endian, a *string* is a length in a
//! byte then as many bytes of UTF-8. The response to a command has the kind of the command with
//! [`frame::RESPONSE`] set, or is an error frame ([`frame::ERROR`]).
//!
//! | Command | Request | Response |
//! | --- | --- | --- |
//! | [`command::INFO`] | nothing | `u8` protocol version, `u32` largest payload, `u32` size of the RAM store, `u32` ticks of the cycle counter per second, `u32` reason of the last reset (as the chip encodes it), `u32` operators whose cycles are kept, `u8` how the previous run ended ([`boot::Crash`]), `u8` crashes in a row before this request, then the strings: firmware, revision of the sources, version of nanoglia, kernel families and target, optimization level, compiler, board, clock, other settings of the board that decide the times, message of the crash; then `u8` number of embedded models and for each a string, its name, and the `u32` length of its bytes |
//! | [`command::WRITE`] | `u32` offset in the RAM store, then the bytes | nothing |
//! | [`command::CRC`] | `u8` store, then `u32` offset and `u32` length in the RAM store, or `u32` index of an embedded model and `u32` its length | `u32` CRC-32 of these bytes (of the whole model) |
//! | [`command::RUN`] | `u8` store of the model, then `u32` its offset and `u32` its length in the RAM store, or `u32` index of an embedded model and `u32` its length; `u32` offset of the arena in the RAM store, `u8` number of inputs, then for each `u32` offset and `u32` length in the RAM store | `u8` [`status`]; if [`status::OK`]: `u32` operators, `u32` graph bytes, `u32` context bytes, `u32` arena bytes used, `u32` peak stack in bytes (`0xFFFFFFFF` if not measured), `u32` cycles of the load, `u32` cycles of the context creation, `u64` cycles of the inference (the sum of those of its operators: what the firmware does between two operators is not counted), `u8` [`flag`]s, `u8` number of outputs, then the `u32` length of each; otherwise `u8` error code of nanoglia ([`error_number`]) and a message |
//! | [`command::OPERATORS`] | `u32` first operator | `u32` first operator, `u8` count, then for each `u32` cycles and a string, its kernel: as many as fit |
//! | [`command::OUTPUT`] | `u8` output, `u32` offset, `u32` length wanted | the bytes, at most a payload |
//! | [`command::RESET`] | `u8` where to: [`Reset`] | nothing, then the board resets |
//! | [`command::FAULT`] | `u8` which: [`Fault`] | nothing, then the board takes the fault |
//!
//! `OPERATORS` and `OUTPUT` read the results of the last `RUN`, which stay in the arena until any
//! other command.
//!
//! No command writes to flash: nothing a host sends can touch the firmware or the bootloader.
//!
//! # Getting out of trouble
//!
//! A frame cut short is dropped after [`FRAME_TIMEOUT_MS`] without a byte. A crash is noted in a
//! byte that survives the reset, by the handler that takes it, and reported by the next `INFO`
//! ([`boot`], [`crash`]); after [`boot::MAX_FAILED_BOOTS`] crashes without a request in between,
//! the handler resets into the bootloader.

#![no_std]

pub mod board;
pub mod boot;
pub mod crash;
pub mod frame;
#[cfg(all(target_arch = "arm", target_os = "none"))]
pub mod stack;

pub use board::{Board, BoardInfo, EmbeddedModel, Fault, LinkError, Reset};

use core::fmt::{self, Write};
use core::ops::{ControlFlow, Range};
use frame::{ErrorCode, HEADER, MAX_FRAME, MAX_PAYLOAD, Reader, TRAILER, Writer};
use nanoglia::{Arena, Encoding, Graph, Observer, OpInfo, Target};

/// The commands of the host.
pub mod command {
    /// What the firmware is.
    pub const INFO: u8 = 1;
    /// Writes bytes into the RAM store.
    pub const WRITE: u8 = 2;
    /// The CRC-32 of bytes of the RAM store, or of an embedded model.
    pub const CRC: u8 = 3;
    /// Loads a model and runs it once.
    pub const RUN: u8 = 4;
    /// The cycles and the kernel of each operator of the last run.
    pub const OPERATORS: u8 = 5;
    /// Bytes of an output of the last run.
    pub const OUTPUT: u8 = 6;
    /// Resets the board.
    pub const RESET: u8 = 7;
    /// Takes a fault on purpose.
    pub const FAULT: u8 = 8;
}

/// Where a model is.
pub mod store {
    /// The block of RAM, written by the host.
    pub const RAM: u8 = 0;
    /// The models linked into the firmware, in flash.
    pub const EMBEDDED: u8 = 1;
}

/// How a `RUN` ended.
pub mod status {
    /// The model ran; the results can be read.
    pub const OK: u8 = 0;
    /// `Graph::load_in` failed.
    pub const LOAD: u8 = 1;
    /// `Graph::init_execution_context_in` failed.
    pub const CONTEXT: u8 = 2;
    /// `set_input` failed.
    pub const INPUT: u8 = 3;
    /// `compute_with` failed.
    pub const COMPUTE: u8 = 4;
    /// The model has more than [`MAX_OUTPUTS`](super::MAX_OUTPUTS) outputs.
    pub const OUTPUTS: u8 = 5;
}

/// Flags of the response to `RUN`.
pub mod flag {
    /// The model has more than [`MAX_OPERATORS`](super::MAX_OPERATORS) operators: the cycles of
    /// the others were not kept and read as 0.
    pub const CYCLES_TRUNCATED: u8 = 1;
}

/// Operators whose cycles are kept.
pub const MAX_OPERATORS: usize = 256;
/// Inputs a case may have.
pub const MAX_INPUTS: usize = 16;
/// Outputs a case may have.
pub const MAX_OUTPUTS: usize = 16;
/// Time without a byte after which a frame cut short is dropped, in milliseconds.
pub const FRAME_TIMEOUT_MS: u32 = 1000;
/// Peak stack of a board that does not measure it.
pub const STACK_UNKNOWN: u32 = u32::MAX;

/// The version of nanoglia the runner is built with.
pub const NANOGLIA_VERSION: &str = env!("RUNNER_NANOGLIA_VERSION");
/// The revision of the sources the runner is built from.
pub const REVISION: &str = env!("RUNNER_REVISION");
/// The kernel families and the target of the build.
pub const BUILD: &str = env!("RUNNER_BUILD");
/// The optimization level of the build.
pub const OPT_LEVEL: &str = env!("RUNNER_OPT_LEVEL");
/// The compiler of the build.
pub const RUSTC: &str = env!("RUNNER_RUSTC");

/// The number of an error code of nanoglia in the response to a failed `RUN`.
pub fn error_number(code: nanoglia::ErrorCode) -> u8 {
    use nanoglia::ErrorCode as Code;
    match code {
        Code::InvalidArgument => 0,
        Code::InvalidEncoding => 1,
        Code::Timeout => 2,
        Code::RuntimeError => 3,
        Code::UnsupportedOperation => 4,
        Code::TooLarge => 5,
        Code::NotFound => 6,
        Code::Security => 7,
        _ => 255,
    }
}

/// A request whose payload is in [`Link::rx`].
#[derive(Clone, Copy)]
struct Request {
    kind: u8,
    seq: u8,
    len: usize,
}

/// The two buffers of the link and what goes through them.
struct Link {
    rx: [u8; MAX_PAYLOAD],
    tx: [u8; MAX_FRAME],
    /// A request read by the loop that serves the results of a run, which was not for it.
    pending: Option<Request>,
}

/// What stopped the reading of a frame.
enum Broken {
    Link(LinkError),
    Frame(ErrorCode, u8),
}

impl From<LinkError> for Broken {
    fn from(error: LinkError) -> Self {
        Self::Link(error)
    }
}

/// Fills `buffer` from the link; a pause of [`FRAME_TIMEOUT_MS`] is a timeout.
fn read_exact<B: Board>(board: &mut B, buffer: &mut [u8]) -> Result<(), LinkError> {
    let mut filled = 0;
    while let Some(rest) = buffer.get_mut(filled..).filter(|rest| !rest.is_empty()) {
        // A read of 0 bytes would loop forever: a board returns at least one.
        filled += board.read(rest, Some(FRAME_TIMEOUT_MS))?.max(1);
    }
    Ok(())
}

impl Link {
    const fn new() -> Self {
        Self {
            rx: [0; MAX_PAYLOAD],
            tx: [0; MAX_FRAME],
            pending: None,
        }
    }

    /// Reads one frame: its payload lands in `rx`.
    fn read_frame<B: Board>(&mut self, board: &mut B) -> Result<Request, Broken> {
        // Wait for the start of a frame, as long as it takes: anything else is noise.
        let mut byte = [0u8; 1];
        let mut previous = 0;
        loop {
            board.read(&mut byte, None)?;
            if previous == frame::SOF[0] && byte[0] == frame::SOF[1] {
                break;
            }
            previous = byte[0];
        }
        let mut header = [0u8; HEADER - 2];
        read_exact(board, &mut header)?;
        let [version, kind, seq, len_low, len_high] = header;
        let len = usize::from(u16::from_le_bytes([len_low, len_high]));
        let Some(payload) = self.rx.get_mut(..len) else {
            return Err(Broken::Frame(ErrorCode::BadFrame, seq));
        };
        read_exact(board, payload)?;
        let mut trailer = [0u8; TRAILER];
        read_exact(board, &mut trailer)?;
        let crc = frame::crc_update(frame::crc_update(frame::CRC_INIT, &header), payload);
        if frame::crc_finish(crc) != u32::from_le_bytes(trailer) {
            return Err(Broken::Frame(ErrorCode::BadCrc, seq));
        }
        if version != frame::VERSION {
            return Err(Broken::Frame(ErrorCode::Version, seq));
        }
        Ok(Request { kind, seq, len })
    }

    /// The next request, whatever it takes: a broken frame gets an error frame and the wait
    /// goes on. Only a closed link ends it.
    fn receive<B: Board>(&mut self, board: &mut B) -> Result<Request, LinkError> {
        if let Some(request) = self.pending.take() {
            return Ok(request);
        }
        loop {
            match self.read_frame(board) {
                Ok(request) => return Ok(request),
                Err(Broken::Link(LinkError::Closed)) => return Err(LinkError::Closed),
                Err(Broken::Link(LinkError::Timeout)) => {
                    self.error(
                        board,
                        0,
                        ErrorCode::Timeout,
                        format_args!("frame cut short"),
                    );
                }
                Err(Broken::Frame(code, seq)) => {
                    self.error(board, seq, code, format_args!("{code:?}"));
                }
            }
        }
    }

    /// The payload of the frame to send.
    fn payload(&mut self) -> Writer<'_> {
        Writer::new(&mut self.tx[HEADER..HEADER + MAX_PAYLOAD])
    }

    /// Sends the frame whose `len` bytes of payload are written. A host that does not read it
    /// loses it.
    fn send<B: Board>(&mut self, board: &mut B, kind: u8, seq: u8, len: usize) {
        if let Some(end) = frame::seal(&mut self.tx, kind, seq, len) {
            let _ = board.write(&self.tx[..end]);
        }
    }

    /// Sends the response to `request`, whose payload `fill` writes.
    fn respond<B: Board>(
        &mut self,
        board: &mut B,
        request: Request,
        fill: impl FnOnce(&mut Writer<'_>),
    ) {
        let mut writer = self.payload();
        fill(&mut writer);
        match writer.finish() {
            Some(len) => self.send(board, request.kind | frame::RESPONSE, request.seq, len),
            None => self.error(
                board,
                request.seq,
                ErrorCode::BadArgument,
                format_args!("the response does not fit in a frame"),
            ),
        }
    }

    /// Sends an error frame.
    fn error<B: Board>(
        &mut self,
        board: &mut B,
        seq: u8,
        code: ErrorCode,
        message: fmt::Arguments<'_>,
    ) {
        let mut writer = self.payload();
        writer.u8(code as u8);
        let _ = writer.write_fmt(message);
        let len = writer.finish().unwrap_or(1);
        self.send(board, frame::ERROR, seq, len);
    }
}

/// `offset..offset + len`, if it does not overflow.
fn span(offset: usize, len: usize) -> Option<Range<usize>> {
    Some(offset..offset.checked_add(len)?)
}

/// The request of a `RUN`.
struct Case {
    store: u8,
    model: Range<usize>,
    arena: usize,
    inputs: [Range<usize>; MAX_INPUTS],
    input_count: usize,
}

impl Case {
    fn parse(payload: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(payload);
        let store = reader.u8()?;
        let model = span(reader.size()?, reader.size()?)?;
        let arena = reader.size()?;
        let input_count = usize::from(reader.u8()?);
        let mut inputs = [const { 0..0 }; MAX_INPUTS];
        for input in inputs.get_mut(..input_count)? {
            *input = span(reader.size()?, reader.size()?)?;
        }
        reader.is_empty().then_some(Self {
            store,
            model,
            arena,
            inputs,
            input_count,
        })
    }
}

/// Times each operator, and tells the board the run is alive.
struct Timer<'a, B: Board> {
    board: &'a mut B,
    cycles: &'a mut [u32; MAX_OPERATORS],
    start: u32,
    /// The sum of the cycles of the operators: a count of 32 bits wraps around after 67 s at
    /// 64 MHz, which an operator never takes (the watchdog ends it first) but a model could.
    total: u64,
    truncated: bool,
}

impl<B: Board> Observer for Timer<'_, B> {
    fn before(&mut self, _op: &OpInfo<'_>) -> ControlFlow<()> {
        self.board.idle();
        self.start = self.board.cycles();
        ControlFlow::Continue(())
    }

    fn after(&mut self, op: &OpInfo<'_>, _result: &nanoglia::Result<()>) {
        let elapsed = self.board.cycles().wrapping_sub(self.start);
        self.total += u64::from(elapsed);
        match self.cycles.get_mut(op.index()) {
            Some(slot) => *slot = elapsed,
            None => self.truncated = true,
        }
    }
}

/// Answers a `RUN` that failed at the step `status`.
fn run_failed<B: Board>(
    link: &mut Link,
    board: &mut B,
    request: Request,
    status: u8,
    error: nanoglia::Error,
) {
    link.respond(board, request, |w| {
        w.u8(status).u8(error_number(error.code));
        let _ = write!(w, "{error}");
    });
}

/// Loads the model of a `RUN` and runs it, answers, then serves `OPERATORS` and `OUTPUT` from the
/// results until another command comes, which is left in [`Link::pending`].
fn run<B: Board>(
    board: &mut B,
    link: &mut Link,
    ram: &mut [u8],
    models: &[EmbeddedModel],
    cycles: &mut [u32; MAX_OPERATORS],
    request: Request,
) {
    let bad = |link: &mut Link, board: &mut B, request: Request, what: &str| {
        link.error(
            board,
            request.seq,
            ErrorCode::BadArgument,
            format_args!("{what}"),
        );
    };
    let Some(case) = link.rx.get(..request.len).and_then(Case::parse) else {
        return bad(link, board, request, "malformed RUN");
    };
    let Some((data, arena_memory)) = ram.split_at_mut_checked(case.arena) else {
        return bad(link, board, request, "the arena starts after the RAM store");
    };
    let data = &*data;
    let model = match case.store {
        store::RAM => data.get(case.model.clone()),
        // The range of an embedded model is its index and its length, which the host knows from
        // the file: a firmware built with another model is caught here.
        store::EMBEDDED => models
            .get(case.model.start)
            .map(|model| model.bytes)
            .filter(|bytes| case.model.start.checked_add(bytes.len()) == Some(case.model.end)),
        _ => None,
    };
    let Some(model) = model else {
        return bad(
            link,
            board,
            request,
            "the model is outside the RAM store or in the arena, or no such embedded model",
        );
    };
    let inputs = &case.inputs[..case.input_count];
    if inputs.iter().any(|input| data.get(input.clone()).is_none()) {
        return bad(
            link,
            board,
            request,
            "an input is outside the RAM store, or in the arena",
        );
    }

    board.stack_paint();
    let arena = Arena::new(arena_memory);
    let start = board.cycles();
    let graph = match Graph::load_in(&[model], Encoding::Autodetect, Target::Auto, &arena) {
        Ok(graph) => graph,
        Err(error) => return run_failed(link, board, request, status::LOAD, error),
    };
    let load_cycles = board.cycles().wrapping_sub(start);
    let start = board.cycles();
    let mut context = match graph.init_execution_context_in(&arena) {
        Ok(context) => context,
        Err(error) => return run_failed(link, board, request, status::CONTEXT, error),
    };
    let context_cycles = board.cycles().wrapping_sub(start);
    for (index, input) in inputs.iter().enumerate() {
        if let Err(error) = context.set_input(index, &data[input.clone()]) {
            return run_failed(link, board, request, status::INPUT, error);
        }
    }
    let output_count = graph.output_count();
    if output_count > MAX_OUTPUTS {
        let error = nanoglia::Error::new(nanoglia::ErrorCode::TooLarge, nanoglia::Detail::None);
        return run_failed(link, board, request, status::OUTPUTS, error);
    }

    cycles.fill(0);
    let mut timer = Timer {
        board: &mut *board,
        cycles: &mut *cycles,
        start: 0,
        total: 0,
        truncated: false,
    };
    let result = context.compute_with(&mut timer);
    let (compute_cycles, truncated) = (timer.total, timer.truncated);
    if let Err(error) = result {
        return run_failed(link, board, request, status::COMPUTE, error);
    }
    let stack_peak = board.stack_peak().unwrap_or(STACK_UNKNOWN);

    let operator_count = graph.operator_count();
    link.respond(board, request, |w| {
        w.u8(status::OK)
            .size(operator_count)
            .size(graph.graph_bytes())
            .size(graph.context_bytes())
            .size(arena.used())
            .u32(stack_peak)
            .u32(load_cycles)
            .u32(context_cycles)
            .bytes(&compute_cycles.to_le_bytes())
            .u8(if truncated { flag::CYCLES_TRUNCATED } else { 0 })
            .u8(output_count as u8);
        for index in 0..output_count {
            w.size(
                context
                    .output(index)
                    .map_or(0, |output| output.bytes().len()),
            );
        }
    });

    // The results stay in the arena: serve them until the host moves on.
    while let Ok(request) = link.receive(board) {
        let mut reader = Reader::new(&link.rx[..request.len]);
        match request.kind {
            command::OPERATORS => {
                let Some(first) = reader.size() else {
                    bad(link, board, request, "malformed OPERATORS");
                    continue;
                };
                link.respond(board, request, |w| {
                    w.size(first);
                    // The count comes before the operators: write them into what is left of the
                    // payload first.
                    let mut count = 0u8;
                    let mut written = 0;
                    let mut rest = [0u8; MAX_PAYLOAD];
                    let mut operators = Writer::new(&mut rest[..w.remaining() - 1]);
                    for index in first..operator_count {
                        let kernel = graph.kernel_name(index).unwrap_or("?");
                        if count == u8::MAX || operators.remaining() < 4 + 1 + kernel.len().min(255)
                        {
                            break;
                        }
                        operators
                            .u32(cycles.get(index).copied().unwrap_or(0))
                            .string(kernel);
                        written = operators.finish().unwrap_or(written);
                        count += 1;
                    }
                    w.u8(count).bytes(&rest[..written]);
                });
            }
            command::OUTPUT => {
                let fields = (|| Some((reader.u8()?, reader.size()?, reader.size()?)))();
                let bytes = fields.and_then(|(index, offset, len)| {
                    let output = context.output(usize::from(index)).ok()?;
                    let bytes = output.bytes().get(offset..)?;
                    Some(&bytes[..bytes.len().min(len).min(MAX_PAYLOAD)])
                });
                match bytes {
                    Some(bytes) => link.respond(board, request, |w| {
                        w.bytes(bytes);
                    }),
                    None => bad(
                        link,
                        board,
                        request,
                        "no such output, or an offset past its end",
                    ),
                }
            }
            _ => {
                link.pending = Some(request);
                return;
            }
        }
    }
}

/// The runner of a board.
pub struct Runner<'a, B: Board> {
    board: B,
    ram: &'a mut [u8],
    models: &'a [EmbeddedModel],
    link: Link,
    cycles: [u32; MAX_OPERATORS],
    boot: boot::Boot,
    crash_message: [u8; crash::MAX_MESSAGE],
    crash_len: usize,
    contacted: bool,
}

impl<'a, B: Board> Runner<'a, B> {
    /// Starts the runner of `board`, with `ram` as its RAM store and the models linked into the
    /// firmware.
    ///
    /// Reads how the previous run ended, for `INFO`.
    pub fn new(mut board: B, ram: &'a mut [u8], models: &'a [EmbeddedModel]) -> Self {
        let mut crash_message = [0; crash::MAX_MESSAGE];
        let crash_len = crash::take(&mut crash_message);
        let boot = boot::start(&mut board);
        Self {
            board,
            ram,
            models,
            link: Link::new(),
            cycles: [0; MAX_OPERATORS],
            boot,
            crash_message,
            crash_len,
            contacted: false,
        }
    }

    /// The board, for the tests.
    pub fn board(&mut self) -> &mut B {
        &mut self.board
    }

    /// Serves the host. Returns only if the link closes or a reset returns, which no board does.
    pub fn serve(&mut self) {
        while let Ok(request) = self.link.receive(&mut self.board) {
            if !self.contacted {
                boot::contact(&mut self.board);
                self.contacted = true;
            }
            if self.handle(request).is_break() {
                return;
            }
        }
    }

    fn handle(&mut self, request: Request) -> ControlFlow<()> {
        let Self {
            board, ram, link, ..
        } = self;
        let models = self.models;
        let mut reader = Reader::new(&link.rx[..request.len]);
        let bad = |link: &mut Link, board: &mut B, what: &str| {
            link.error(
                board,
                request.seq,
                ErrorCode::BadArgument,
                format_args!("{what}"),
            );
            ControlFlow::Continue(())
        };
        match request.kind {
            command::INFO => {
                let info = board.info();
                let message =
                    core::str::from_utf8(&self.crash_message[..self.crash_len]).unwrap_or("");
                let boot = self.boot;
                link.respond(board, request, |w| {
                    w.u8(frame::VERSION)
                        .size(MAX_PAYLOAD)
                        .size(ram.len())
                        .u32(info.counter_hz)
                        .u32(info.reset_reason)
                        .size(MAX_OPERATORS)
                        .u8(boot.crash as u8)
                        .u8(boot.failed)
                        .string(info.firmware)
                        .string(REVISION)
                        .string(NANOGLIA_VERSION)
                        .string(BUILD)
                        .string(OPT_LEVEL)
                        .string(RUSTC)
                        .string(info.board)
                        .string(info.clock)
                        .string(info.settings)
                        .string(message)
                        .u8(models.len().min(255) as u8);
                    for model in models.iter().take(255) {
                        w.string(model.name).size(model.bytes.len());
                    }
                });
            }
            command::WRITE => {
                let Some(offset) = reader.size() else {
                    return bad(link, board, "malformed WRITE");
                };
                let data = reader.rest();
                match span(offset, data.len()).and_then(|range| ram.get_mut(range)) {
                    Some(slot) => slot.copy_from_slice(data),
                    None => return bad(link, board, "outside the RAM store"),
                }
                link.respond(board, request, |_| {});
            }
            command::CRC => {
                let (Some(store), Some(offset), Some(len)) =
                    (reader.u8(), reader.size(), reader.size())
                else {
                    return bad(link, board, "malformed CRC");
                };
                let bytes = match store {
                    store::RAM => span(offset, len).and_then(|range| ram.get(range)),
                    store::EMBEDDED => models
                        .get(offset)
                        .map(|model| model.bytes)
                        .filter(|bytes| bytes.len() == len),
                    _ => None,
                };
                let Some(bytes) = bytes else {
                    return bad(
                        link,
                        board,
                        "outside the RAM store, or no such embedded model",
                    );
                };
                let crc = frame::crc32(bytes);
                link.respond(board, request, |w| {
                    w.u32(crc);
                });
            }
            command::RUN => run(board, link, ram, models, &mut self.cycles, request),
            command::OPERATORS | command::OUTPUT => {
                let message = format_args!("no results: RUN first");
                link.error(board, request.seq, ErrorCode::NoRun, message);
            }
            command::RESET => {
                let to = match reader.u8() {
                    Some(0) => Reset::Application,
                    Some(1) => Reset::Bootloader,
                    _ => return bad(link, board, "malformed RESET"),
                };
                link.respond(board, request, |_| {});
                board.reset(to);
                return ControlFlow::Break(());
            }
            command::FAULT => {
                let fault = match reader.u8() {
                    Some(1) => Fault::Panic,
                    Some(2) => Fault::HardFault,
                    Some(3) => Fault::Hang,
                    _ => return bad(link, board, "malformed FAULT"),
                };
                link.respond(board, request, |_| {});
                board.fault(fault);
                return ControlFlow::Break(());
            }
            kind => {
                let message = format_args!("unknown command {kind}");
                link.error(board, request.seq, ErrorCode::UnknownCommand, message);
            }
        }
        ControlFlow::Continue(())
    }
}
