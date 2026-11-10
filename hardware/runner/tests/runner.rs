//! The runner on the host, behind a link held in memory: whole golden cases, with the model in RAM
//! and linked in, broken frames, pauses, bad arguments, and the way out of crashes.

use nanoglia_hardware_runner::frame::{self, ErrorCode, HEADER, MAX_PAYLOAD, TRAILER};
use nanoglia_hardware_runner::{
    Board, BoardInfo, EmbeddedModel, LinkError, Reset, Runner, boot, command, flag, include_model,
    status, store,
};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// What the host sends: bytes, and pauses longer than the frame timeout.
enum Sent {
    Byte(u8),
    Pause,
}

/// A board whose link is two queues.
#[derive(Default)]
struct MemoryBoard {
    input: VecDeque<Sent>,
    output: Vec<u8>,
    clock: u32,
    retained: u8,
    resets: Vec<Reset>,
    idles: usize,
    contacts: usize,
}

impl Board for MemoryBoard {
    fn info(&self) -> BoardInfo {
        BoardInfo {
            board: "memory",
            firmware: "test 0.0.0",
            clock: "a counter",
            counter_hz: 1000,
            reset_reason: 4,
            settings: "no cache",
        }
    }

    fn read(&mut self, buffer: &mut [u8], timeout_ms: Option<u32>) -> Result<usize, LinkError> {
        let mut read = 0;
        while read < buffer.len() {
            match self.input.front() {
                Some(Sent::Byte(byte)) => {
                    buffer[read] = *byte;
                    read += 1;
                    self.input.pop_front();
                }
                // A pause ends what was read before it; alone, it is a timeout for a read that
                // has one, and nothing for a read that waits forever.
                Some(Sent::Pause) if read > 0 => break,
                Some(Sent::Pause) => {
                    self.input.pop_front();
                    if timeout_ms.is_some() {
                        return Err(LinkError::Timeout);
                    }
                }
                None if read > 0 => break,
                None => return Err(LinkError::Closed),
            }
        }
        Ok(read)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), LinkError> {
        self.output.extend_from_slice(bytes);
        Ok(())
    }

    fn cycles(&mut self) -> u32 {
        self.clock += 7;
        self.clock
    }

    fn idle(&mut self) {
        self.idles += 1;
    }

    fn reset(&mut self, to: Reset) {
        self.resets.push(to);
    }

    fn contact(&mut self) {
        self.contacts += 1;
    }

    fn retained(&mut self) -> u8 {
        self.retained
    }

    fn set_retained(&mut self, value: u8) {
        self.retained = value;
    }

    fn stack_peak(&mut self) -> Option<u32> {
        Some(1234)
    }
}

/// The models linked into the runner of the tests.
static MODELS: [EmbeddedModel; 2] = [
    EmbeddedModel {
        name: "classifier.tflite",
        bytes: include_model!("../../../crates/nanoglia/doc/classifier.tflite"),
    },
    EmbeddedModel {
        name: "kws_ref_model.tflite",
        bytes: include_model!("../../../testdata/mlperf-tiny/kws_ref_model.tflite"),
    },
];

/// A frame of the host.
fn request(kind: u8, seq: u8, payload: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; HEADER + payload.len() + TRAILER];
    bytes[HEADER..HEADER + payload.len()].copy_from_slice(payload);
    let len = frame::seal(&mut bytes, kind, seq, payload.len()).unwrap();
    assert_eq!(len, bytes.len());
    bytes
}

/// A frame of the device.
#[derive(Debug)]
struct Response {
    kind: u8,
    seq: u8,
    payload: Vec<u8>,
}

impl Response {
    fn error(&self) -> Option<u8> {
        (self.kind == frame::ERROR).then(|| self.payload[0])
    }

    fn u32(&self, offset: usize) -> u32 {
        u32::from_le_bytes(self.payload[offset..offset + 4].try_into().unwrap())
    }
}

/// Splits what the device wrote into frames, checking each.
fn responses(mut bytes: &[u8]) -> Vec<Response> {
    let mut frames = Vec::new();
    while !bytes.is_empty() {
        assert_eq!(&bytes[..2], b"NG");
        assert_eq!(bytes[2], frame::VERSION);
        let len = usize::from(u16::from_le_bytes([bytes[5], bytes[6]]));
        let end = HEADER + len;
        let crc = u32::from_le_bytes(bytes[end..end + TRAILER].try_into().unwrap());
        assert_eq!(crc, frame::crc32(&bytes[2..end]));
        frames.push(Response {
            kind: bytes[3],
            seq: bytes[4],
            payload: bytes[HEADER..end].to_vec(),
        });
        bytes = &bytes[end + TRAILER..];
    }
    frames
}

/// Runs the runner on what the host sends, with `ram` bytes of RAM store, and returns what it
/// answered.
fn exchange(sent: Vec<Sent>, ram: usize) -> (Vec<Response>, MemoryBoard) {
    exchange_on(MemoryBoard::default(), sent, ram)
}

fn exchange_on(
    mut board: MemoryBoard,
    sent: Vec<Sent>,
    ram: usize,
) -> (Vec<Response>, MemoryBoard) {
    board.input = sent.into();
    let mut ram = vec![0u8; ram];
    let mut runner = Runner::new(board, &mut ram, &MODELS);
    runner.serve();
    let board = std::mem::take(runner.board());
    (responses(&board.output), board)
}

fn bytes(frames: &[Vec<u8>]) -> Vec<Sent> {
    frames
        .iter()
        .flatten()
        .map(|&byte| Sent::Byte(byte))
        .collect()
}

fn le(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

/// The model, the inputs and the expected outputs of a golden case with one input and one output.
fn case(name: &str, model: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let directory = testdata().join(name);
    (
        std::fs::read(testdata().join(model)).unwrap(),
        std::fs::read(directory.join("input_0.bin")).unwrap(),
        std::fs::read(directory.join("output_0.bin")).unwrap(),
    )
}

/// `WRITE` requests of `data` at `offset` of the RAM store, in payloads of at most `chunk` bytes.
fn writes(seq: &mut u8, offset: usize, data: &[u8], chunk: usize) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    for (index, part) in data.chunks(chunk).enumerate() {
        let mut payload = ((offset + index * chunk) as u32).to_le_bytes().to_vec();
        payload.extend(part);
        *seq = seq.wrapping_add(1);
        frames.push(request(command::WRITE, *seq, &payload));
    }
    frames
}

const KWS: (&str, &str) = (
    "mlperf-tiny/golden/kws_ref_model/sample_0",
    "mlperf-tiny/kws_ref_model.tflite",
);

/// The `count` strings at the start of `bytes`, and what follows them.
fn strings(mut bytes: &[u8], count: usize) -> (Vec<String>, &[u8]) {
    let mut strings = Vec::new();
    for _ in 0..count {
        let (&len, rest) = bytes.split_first().unwrap();
        strings.push(String::from_utf8(rest[..usize::from(len)].to_vec()).unwrap());
        bytes = &rest[usize::from(len)..];
    }
    (strings, bytes)
}

#[test]
fn info_describes_the_firmware() {
    let (frames, board) = exchange(bytes(&[request(command::INFO, 9, &[])]), 4096);
    let [info] = &frames[..] else {
        panic!("{frames:?}")
    };
    assert_eq!((info.kind, info.seq), (command::INFO | frame::RESPONSE, 9));
    assert_eq!(info.payload[0], frame::VERSION);
    assert_eq!(info.u32(1) as usize, MAX_PAYLOAD);
    assert_eq!(info.u32(5), 4096);
    assert_eq!(info.u32(9), 1000);
    assert_eq!(info.u32(13), 4);
    assert_eq!(info.u32(17), 256);
    assert_eq!(&info.payload[21..23], &[0, 0]);
    let (strings, models) = strings(&info.payload[23..], 10);
    assert_eq!(strings[0], "test 0.0.0");
    assert!(!strings[1].is_empty(), "revision");
    assert_eq!(
        strings[2].split('.').count(),
        3,
        "version of nanoglia: {}",
        strings[2]
    );
    assert!(
        strings[3].starts_with("cmsis-nn kernels float int16 all-ops, "),
        "{}",
        strings[3]
    );
    assert!(
        ["0", "1", "2", "3", "s", "z"].contains(&&*strings[4]),
        "optimization level: {}",
        strings[4]
    );
    assert!(strings[5].starts_with("rustc "), "{}", strings[5]);
    assert_eq!(&strings[6..], ["memory", "a counter", "no cache", ""]);
    // The embedded models: a count, then a name and a length each.
    assert_eq!(models[0], 2);
    let (names, rest) = self::strings(&models[1..], 1);
    assert_eq!(names, ["classifier.tflite"]);
    assert_eq!(
        u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize,
        MODELS[0].bytes.len()
    );
    let (names, rest) = self::strings(&rest[4..], 1);
    assert_eq!(names, ["kws_ref_model.tflite"]);
    assert_eq!(rest, (MODELS[1].bytes.len() as u32).to_le_bytes());
    // The request cleared the count of failed boots (there was none) and left no reset.
    assert_eq!(boot::decode(board.retained), (boot::Crash::None, 0));
    assert!(board.resets.is_empty());
    assert_eq!(board.contacts, 1);
}

/// Sends a golden case, runs it and reads everything back.
fn run_case(model_store: u8) {
    let (model, input, expected) = case(KWS.0, KWS.1);
    let mut seq = 0;
    let mut frames = Vec::new();
    // Where the model is: in the RAM store, or the second model linked in.
    let (model_at, input_offset) = if model_store == store::EMBEDDED {
        (1, 0)
    } else {
        frames.extend(writes(&mut seq, 0, &model, MAX_PAYLOAD - 4));
        (0, model.len().next_multiple_of(16))
    };
    frames.extend(writes(&mut seq, input_offset, &input, 333));
    let arena = (input_offset + input.len()).next_multiple_of(16);
    // The CRC of the model where it is.
    let mut payload = vec![model_store];
    payload.extend(le(&[model_at, model.len() as u32]));
    frames.push(request(command::CRC, 201, &payload));
    let mut payload = vec![model_store];
    payload.extend(le(&[model_at, model.len() as u32, arena as u32]));
    payload.push(1);
    payload.extend(le(&[input_offset as u32, input.len() as u32]));
    frames.push(request(command::RUN, 202, &payload));
    frames.push(request(command::OPERATORS, 203, &le(&[0])));
    frames.push(request(command::OPERATORS, 204, &le(&[5])));
    let mut payload = vec![0];
    payload.extend(le(&[0, 5]));
    frames.push(request(command::OUTPUT, 205, &payload));
    let mut payload = vec![0];
    payload.extend(le(&[5, 1000]));
    frames.push(request(command::OUTPUT, 206, &payload));
    // Past the end of the output, and no such output.
    let mut payload = vec![0];
    payload.extend(le(&[1000, 1]));
    frames.push(request(command::OUTPUT, 207, &payload));
    let mut payload = vec![3];
    payload.extend(le(&[0, 1]));
    frames.push(request(command::OUTPUT, 208, &payload));
    // Any other command ends the reading of the results.
    frames.push(request(command::INFO, 209, &[]));
    frames.push(request(command::OPERATORS, 210, &le(&[0])));

    let (responses, board) = exchange(bytes(&frames), 128 * 1024);
    assert!(
        responses.iter().all(|r| r.seq > 206 || r.error().is_none()),
        "{responses:?}"
    );
    let by_seq = |seq: u8| responses.iter().find(|r| r.seq == seq).unwrap();
    assert_eq!(by_seq(201).u32(0), frame::crc32(&model));

    let run = by_seq(202);
    assert_eq!(run.kind, command::RUN | frame::RESPONSE);
    assert_eq!(
        run.payload[0],
        status::OK,
        "{:?}",
        String::from_utf8_lossy(&run.payload[2..])
    );
    let operators = run.u32(1) as usize;
    assert_eq!(operators, 13);
    let (graph, context, used) = (run.u32(5), run.u32(9), run.u32(13));
    assert!(
        graph > 0 && context >= 16000 && used >= graph + context,
        "{graph} {context} {used}"
    );
    assert_eq!(run.u32(17), 1234, "peak stack");
    // The clock of the board moves 7 ticks per reading.
    assert_eq!(run.u32(21), 7, "cycles of the load");
    assert_eq!(run.u32(25), 7, "cycles of the context");
    // One reading before and one after each operator: 7 ticks each, and nothing in between.
    let inference = u64::from_le_bytes(run.payload[29..37].try_into().unwrap());
    assert_eq!(inference, 7 * operators as u64, "cycles of the inference");
    assert_eq!(run.payload[37], 0, "flags");
    assert_eq!(run.payload[38], 1, "outputs");
    assert_eq!(run.u32(39) as usize, expected.len());
    assert_eq!(board.idles, operators);

    // The operators, from two requests that overlap.
    let parse = |response: &Response| {
        let first = response.u32(0) as usize;
        let count = usize::from(response.payload[4]);
        let mut rest = &response.payload[5..];
        let mut kernels = Vec::new();
        for _ in 0..count {
            let cycles = u32::from_le_bytes(rest[..4].try_into().unwrap());
            let len = usize::from(rest[4]);
            kernels.push((
                cycles,
                String::from_utf8(rest[5..5 + len].to_vec()).unwrap(),
            ));
            rest = &rest[5 + len..];
        }
        assert!(rest.is_empty());
        (first, kernels)
    };
    let (first, all) = parse(by_seq(203));
    assert_eq!((first, all.len()), (0, operators));
    assert!(all.iter().all(|(cycles, _)| *cycles == 7), "{all:?}");
    assert_eq!(all[0].1, "cmsis-nn/conv2d_s8");
    assert!(
        all.iter()
            .all(|(_, kernel)| kernel.starts_with("cmsis-nn/") || kernel.starts_with("rust/"))
    );
    let (first, tail) = parse(by_seq(204));
    assert_eq!(first, 5);
    assert_eq!(tail, all[5..]);

    let mut output = by_seq(205).payload.clone();
    assert_eq!(output.len(), 5);
    output.extend(&by_seq(206).payload);
    assert_eq!(output, expected, "the output of the model");
    assert_eq!(by_seq(207).error(), Some(ErrorCode::BadArgument as u8));
    assert_eq!(by_seq(208).error(), Some(ErrorCode::BadArgument as u8));
    assert_eq!(by_seq(209).kind, command::INFO | frame::RESPONSE);
    assert_eq!(by_seq(210).error(), Some(ErrorCode::NoRun as u8));
}

#[test]
fn a_case_runs_with_its_model_in_ram() {
    run_case(store::RAM);
}

#[test]
fn a_case_runs_with_its_model_linked_in() {
    run_case(store::EMBEDDED);
}

#[test]
fn broken_frames_get_errors_and_the_next_one_is_served() {
    let info = request(command::INFO, 1, &[]);
    let mut sent = Vec::new();
    // Noise, even with a start of frame in it.
    sent.extend(bytes(&[b"AT\r\nN NNX G".to_vec()]));
    // A wrong CRC.
    let mut broken = request(command::CRC, 2, &[0; 9]);
    *broken.last_mut().unwrap() ^= 0x80;
    sent.extend(bytes(&[broken]));
    // A corrupted payload.
    let mut broken = request(command::CRC, 3, &[0; 9]);
    broken[HEADER + 2] ^= 1;
    sent.extend(bytes(&[broken]));
    // A length no frame has.
    sent.extend(bytes(&[vec![
        b'N',
        b'G',
        frame::VERSION,
        command::WRITE,
        4,
        0xFF,
        0xFF,
    ]]));
    // A frame cut in its header, then in its payload: the host went silent.
    sent.extend(bytes(&[request(command::CRC, 5, &[0; 9])[..4].to_vec()]));
    sent.push(Sent::Pause);
    sent.extend(bytes(&[request(command::CRC, 6, &[0; 9])[..12].to_vec()]));
    sent.push(Sent::Pause);
    // Another version of the protocol.
    let mut other = request(command::INFO, 7, &[]);
    other[2] = frame::VERSION + 1;
    let crc = frame::crc32(&other[2..HEADER]);
    other[HEADER..].copy_from_slice(&crc.to_le_bytes());
    sent.extend(bytes(&[other]));
    // An unknown command, and at last a good frame. A pause between frames is no error.
    sent.extend(bytes(&[request(0x55, 8, &[])]));
    sent.push(Sent::Pause);
    sent.extend(bytes(&[info]));

    let (frames, _) = exchange(sent, 1024);
    let got: Vec<_> = frames.iter().map(|r| (r.seq, r.error())).collect();
    assert_eq!(
        got,
        [
            (2, Some(ErrorCode::BadCrc as u8)),
            (3, Some(ErrorCode::BadCrc as u8)),
            (4, Some(ErrorCode::BadFrame as u8)),
            (0, Some(ErrorCode::Timeout as u8)),
            (0, Some(ErrorCode::Timeout as u8)),
            (7, Some(ErrorCode::Version as u8)),
            (8, Some(ErrorCode::UnknownCommand as u8)),
            (1, None),
        ]
    );
}

#[test]
fn bad_arguments_are_refused() {
    let mut seq = 0;
    let mut frames = Vec::new();
    let mut expected = Vec::new();
    let mut push = |kind: u8, payload: Vec<u8>, error: Option<ErrorCode>| {
        seq += 1;
        frames.push(request(kind, seq, &payload));
        expected.push((seq, error.map(|e| e as u8)));
    };
    let with = |store: u8, values: &[u32], tail: &[u8]| [&[store][..], &le(values), tail].concat();
    let bad = Some(ErrorCode::BadArgument);
    let kws = MODELS[1].bytes.len() as u32;

    // Writes: at the end of the RAM store, one byte past it, an offset that overflows.
    push(command::WRITE, [le(&[1020]), vec![1; 4]].concat(), None);
    push(command::WRITE, [le(&[1021]), vec![1; 4]].concat(), bad);
    push(command::WRITE, [le(&[u32::MAX]), vec![1; 4]].concat(), bad);
    push(command::WRITE, vec![0, 0], bad);
    // CRC: inside the RAM store, or a model the firmware has, with its length.
    push(command::CRC, with(store::RAM, &[0, 1024], &[]), None);
    push(command::CRC, with(store::RAM, &[1, 1024], &[]), bad);
    push(command::CRC, with(store::EMBEDDED, &[1, kws], &[]), None);
    push(command::CRC, with(store::EMBEDDED, &[1, kws - 1], &[]), bad);
    push(command::CRC, with(store::EMBEDDED, &[2, kws], &[]), bad);
    push(command::CRC, with(7, &[0, 16], &[]), bad);
    // RUN: the model and the inputs before the arena, the arena inside the store.
    push(command::RUN, with(store::RAM, &[0, 64, 2000], &[0]), bad);
    push(command::RUN, with(store::RAM, &[0, 64, 32], &[0]), bad);
    push(
        command::RUN,
        with(store::RAM, &[0, 16, 32], &[1, 20, 0, 0, 0, 16, 0, 0, 0]),
        bad,
    );
    push(command::RUN, with(store::RAM, &[0, 16, 32], &[17]), bad);
    push(command::RUN, with(store::RAM, &[0, 16, 32], &[0, 0]), bad);
    push(command::RUN, with(9, &[0, 16, 32], &[0]), bad);
    // No such embedded model, or not the one the host thinks.
    push(
        command::RUN,
        with(store::EMBEDDED, &[2, kws, 32], &[0]),
        bad,
    );
    push(
        command::RUN,
        with(store::EMBEDDED, &[1, kws + 4, 32], &[0]),
        bad,
    );
    push(
        command::RUN,
        with(store::EMBEDDED, &[u32::MAX, kws, 32], &[0]),
        bad,
    );
    // Results without a run, resets and faults that are none.
    push(
        command::OUTPUT,
        with(0, &[0, 1], &[]),
        Some(ErrorCode::NoRun),
    );
    push(command::RESET, vec![2], bad);
    push(command::RESET, vec![], bad);
    push(command::FAULT, vec![0], bad);
    // A command of another version of the protocol.
    push(0x55, vec![], Some(ErrorCode::UnknownCommand));

    let (responses, board) = exchange(bytes(&frames), 1024);
    let got: Vec<_> = responses.iter().map(|r| (r.seq, r.error())).collect();
    assert_eq!(got, expected);
    assert!(board.resets.is_empty());
}

#[test]
fn a_model_that_does_not_fit_or_is_no_model_fails_to_load() {
    let (model, input, _) = case(KWS.0, KWS.1);
    let mut seq = 0;
    let mut frames = writes(&mut seq, 0, &model, 1000);
    let input_offset = model.len().next_multiple_of(16);
    frames.extend(writes(&mut seq, input_offset, &input, 1000));
    let arena = (input_offset + input.len()).next_multiple_of(16);
    let mut payload = vec![store::RAM];
    payload.extend(le(&[0, model.len() as u32, arena as u32]));
    payload.push(1);
    payload.extend(le(&[input_offset as u32, input.len() as u32]));
    frames.push(request(command::RUN, 100, &payload));
    // The same bytes are no model from their second byte.
    let mut payload = vec![store::RAM];
    payload.extend(le(&[16, 4096, arena as u32]));
    payload.push(0);
    frames.push(request(command::RUN, 101, &payload));
    // An input of the wrong size.
    let mut payload = vec![store::RAM];
    payload.extend(le(&[0, model.len() as u32, arena as u32]));
    payload.push(1);
    payload.extend(le(&[input_offset as u32, input.len() as u32 - 1]));
    frames.push(request(command::RUN, 102, &payload));

    // 4 KiB of arena: the graph alone needs about 8.
    let (responses, _) = exchange(bytes(&frames), arena + 4 * 1024);
    let run = responses.iter().find(|r| r.seq == 100).unwrap();
    assert_eq!(run.kind, command::RUN | frame::RESPONSE);
    assert_eq!(run.payload[0], status::LOAD);
    assert_eq!(run.payload[1], 5, "too large");
    let message = String::from_utf8(run.payload[2..].to_vec()).unwrap();
    assert!(message.starts_with("too large"), "{message}");

    let (responses, _) = exchange(bytes(&frames), arena + 64 * 1024);
    let by_seq = |seq: u8| responses.iter().find(|r| r.seq == seq).unwrap();
    assert_eq!(by_seq(100).payload[0], status::OK);
    assert_eq!(by_seq(100).payload[37] & flag::CYCLES_TRUNCATED, 0);
    assert_eq!(
        &by_seq(101).payload[..2],
        &[status::LOAD, 1],
        "invalid encoding"
    );
    assert_eq!(
        &by_seq(102).payload[..2],
        &[status::INPUT, 0],
        "invalid argument"
    );
}

#[test]
fn reset_answers_then_resets() {
    let frames = [
        request(command::RESET, 1, &[1]),
        // Never read: a board is gone by then.
        request(command::INFO, 2, &[]),
    ];
    let (responses, board) = exchange(bytes(&frames), 64);
    let got: Vec<_> = responses.iter().map(|r| (r.kind, r.seq)).collect();
    assert_eq!(got, [(command::RESET | frame::RESPONSE, 1)]);
    assert_eq!(board.resets, [Reset::Bootloader]);

    let (_, board) = exchange(bytes(&[request(command::RESET, 1, &[0])]), 64);
    assert_eq!(board.resets, [Reset::Application]);
}

#[test]
fn crashes_are_reported_and_a_request_clears_their_count() {
    // The previous run panicked: INFO says so, and the request clears the count.
    let board = MemoryBoard {
        retained: boot::crash(0, boot::Crash::Panic).retained,
        ..Default::default()
    };
    let (responses, board) = exchange_on(board, bytes(&[request(command::INFO, 1, &[])]), 64);
    assert_eq!(
        &responses[0].payload[21..23],
        &[boot::Crash::Panic as u8, 1]
    );
    assert_eq!(boot::decode(board.retained), (boot::Crash::None, 0));
    assert!(board.resets.is_empty());

    // Without a request, the runner forgets the kind of the crash it will report and keeps the
    // count: the next crash goes on from it. The runner itself never resets into the bootloader:
    // the handler of the crash does (`boot::crash`), even when the runner never started.
    let twice = boot::crash(
        boot::crash(0, boot::Crash::Fault).retained,
        boot::Crash::Watchdog,
    );
    let board = MemoryBoard {
        retained: twice.retained,
        ..Default::default()
    };
    let (_, board) = exchange_on(board, Vec::new(), 64);
    assert_eq!(boot::decode(board.retained), (boot::Crash::None, 2));
    assert!(board.resets.is_empty() && board.contacts == 0);
    let verdict = boot::crash(board.retained, boot::Crash::Panic);
    assert_eq!(verdict.to, Reset::Bootloader);
    assert_eq!(boot::decode(verdict.retained), (boot::Crash::None, 0));

    // A reset the host asked for is no crash: the count does not move.
    let (_, board) = exchange(Vec::new(), 64);
    assert_eq!(boot::decode(board.retained), (boot::Crash::None, 0));
}
