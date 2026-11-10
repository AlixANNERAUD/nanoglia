"""The protocol of hardware/runner, from the host: frames (runner/src/frame.rs) and commands
(runner/src/lib.rs). Keep the three in step.

A `Link` sends a request and waits for its response over a transport: anything with
`write(data)`, `read(size, timeout) -> bytes` (fewer bytes, or none, when the time is up) and
`close()`. A transport raises `Disconnected` when the device is gone; one whose device runs only
inside `write` and `read` (the emulated and simulated ones) sets `instant`: an empty read then
means that nothing will come, without waiting for the time to pass. A `Device` is the commands
on top of a link.
"""

import struct
import time
import zlib
from dataclasses import dataclass, field

SOF = b"NG"
VERSION = 1
MAX_PAYLOAD = 1024
HEADER = 7
TRAILER = 4
RESPONSE = 0x80
ERROR = 0xFF

# Commands.
INFO, WRITE, CRC, RUN, OPERATORS, OUTPUT, RESET, FAULT = range(1, 9)
# Where a model is.
RAM, EMBEDDED = 0, 1
# Status of a RUN.
OK, LOAD, CONTEXT, INPUT, COMPUTE, OUTPUTS = range(6)
STAGES = {LOAD: "load", CONTEXT: "context", INPUT: "input", COMPUTE: "compute", OUTPUTS: "outputs"}
# Flags of a RUN.
CYCLES_TRUNCATED = 1
# Error code of nanoglia for a model that does not fit in the memory it is given.
TOO_LARGE = 5
# Codes of an error frame.
BAD_FRAME, BAD_CRC, BAD_VERSION, UNKNOWN_COMMAND, BAD_ARGUMENT, FRAME_TIMEOUT, NO_RUN = range(1, 8)
# The errors of the link, which a retry may cure, as opposed to those of the request.
LINK_ERRORS = {BAD_FRAME, BAD_CRC, FRAME_TIMEOUT}
# Where a RESET leads.
APPLICATION, BOOTLOADER = 0, 1
# The faults of FAULT.
PANIC, HARD_FAULT, HANG = 1, 2, 3
# How the previous run ended (boot.rs).
CRASHES = {0: None, 1: "panic", 2: "fault", 3: "watchdog"}
STACK_UNKNOWN = 0xFFFF_FFFF


class Disconnected(Exception):
    """The device is gone: its port closed, or it stopped running."""


class NoAnswer(Exception):
    """The device did not answer a request in time, after every retry."""


class DeviceError(Exception):
    """The device refused a request with an error frame."""

    def __init__(self, code: int, message: str):
        super().__init__(f"error {code}: {message}")
        self.code = code
        self.message = message


class ProtocolError(Exception):
    """The device answered something the protocol does not allow."""


def encode(kind: int, seq: int, payload: bytes = b"") -> bytes:
    """The frame of `kind` and `seq` around `payload`."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload of {len(payload)} bytes, {MAX_PAYLOAD} at most")
    body = bytes([VERSION, kind, seq]) + struct.pack("<H", len(payload)) + payload
    return SOF + body + struct.pack("<I", zlib.crc32(body))


@dataclass
class Frame:
    kind: int
    seq: int
    payload: bytes


class Link:
    """Requests and responses over a transport, with retries."""

    def __init__(self, transport, timeout: float = 2.0, attempts: int = 3):
        self.transport = transport
        self.timeout = timeout
        self.attempts = attempts
        self.seq = 0
        self.buffer = bytearray()
        # Counters, for the report of a run: frames sent again, frames dropped.
        self.retries = 0
        self.dropped = 0

    def close(self) -> None:
        self.transport.close()

    def _fill(self, size: int, deadline: float) -> bool:
        """Reads until the buffer holds `size` bytes; False if the deadline passes first."""
        while len(self.buffer) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            data = self.transport.read(max(size - len(self.buffer), 1), remaining)
            if not data and getattr(self.transport, "instant", False):
                # A device that runs only when the host moves has nothing more to say.
                return False
            self.buffer += data
        return True

    def _frame(self, deadline: float) -> Frame | None:
        """The next well-formed frame, or None at the deadline. Anything else is dropped."""
        while True:
            if not self._fill(HEADER, deadline):
                return None
            start = self.buffer.find(SOF)
            if start != 0:
                # Noise: drop it, up to the next start of frame (or keep a lone last 'N').
                keep = start if start > 0 else len(self.buffer) - (self.buffer[-1:] == SOF[:1])
                del self.buffer[:keep]
                continue
            (length,) = struct.unpack_from("<H", self.buffer, 5)
            if length > MAX_PAYLOAD:
                del self.buffer[:2]
                self.dropped += 1
                continue
            end = HEADER + length + TRAILER
            if not self._fill(end, deadline):
                return None
            body = bytes(self.buffer[2 : HEADER + length])
            (crc,) = struct.unpack_from("<I", self.buffer, HEADER + length)
            if crc != zlib.crc32(body):
                del self.buffer[:2]
                self.dropped += 1
                continue
            del self.buffer[:end]
            if body[0] != VERSION:
                raise ProtocolError(f"the device speaks version {body[0]} of the protocol, the host {VERSION}")
            return Frame(body[1], body[2], body[5:])

    def request(self, kind: int, payload: bytes = b"", timeout: float | None = None, attempts: int | None = None):
        """Sends a request and returns the payload of its response.

        Sent again when no answer comes in time or when the device reports a broken frame; raises
        `NoAnswer` after `attempts` tries, `DeviceError` when the device refuses the request.
        """
        timeout = self.timeout if timeout is None else timeout
        attempts = self.attempts if attempts is None else attempts
        self.seq = (self.seq + 1) % 256
        frame = encode(kind, self.seq, payload)
        problem = "no answer"
        for attempt in range(attempts):
            if attempt:
                self.retries += 1
            # What is left of an earlier exchange is not for this request.
            self.buffer.clear()
            self.transport.write(frame)
            deadline = time.monotonic() + timeout
            while (answer := self._frame(deadline)) is not None:
                if answer.seq != self.seq:
                    # The answer to an earlier request, or an error about noise on the line.
                    self.dropped += 1
                    continue
                if answer.kind == kind | RESPONSE:
                    return answer.payload
                if answer.kind != ERROR or not answer.payload:
                    raise ProtocolError(f"frame of kind {answer.kind:#x} in answer to command {kind}")
                code, message = answer.payload[0], answer.payload[1:].decode(errors="replace")
                if code not in LINK_ERRORS:
                    raise DeviceError(code, message)
                problem = f"the device received a broken frame ({message})"
                break
        raise NoAnswer(f"command {kind}: {problem}, after {attempts} attempt(s) of {timeout:g} s")


class Reader:
    """Reads the fields of a payload in order."""

    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def unpack(self, fmt: str):
        try:
            values = struct.unpack_from("<" + fmt, self.data, self.at)
        except struct.error as error:
            raise ProtocolError(f"payload cut short: {error}") from None
        self.at += struct.calcsize("<" + fmt)
        return values if len(values) > 1 else values[0]

    def string(self) -> str:
        length = self.unpack("B")
        if self.at + length > len(self.data):
            raise ProtocolError("string cut short")
        text = self.data[self.at : self.at + length].decode(errors="replace")
        self.at += length
        return text

    def rest(self) -> bytes:
        data, self.at = self.data[self.at :], len(self.data)
        return data


@dataclass
class Info:
    """The answer to INFO."""

    protocol: int
    max_payload: int
    ram_bytes: int
    counter_hz: int
    reset_reason: int
    max_operators: int
    crash: str | None
    failed_boots: int
    firmware: str
    revision: str
    nanoglia: str
    build: str
    board: str
    clock: str
    crash_message: str
    # Name and length of each model linked into the firmware, in the order of their index.
    models: list[tuple[str, int]] = field(default_factory=list)

    def describe_crash(self) -> str:
        """How the previous run of the firmware ended, in words; empty if it did not crash."""
        if self.crash is None:
            return ""
        return f"the firmware restarted after a {self.crash}" + (f": {self.crash_message}" if self.crash_message else "")


@dataclass
class RunResult:
    """The answer to RUN."""

    status: int
    # When the run failed: the error code of nanoglia and its message.
    error: int = 0
    message: str = ""
    operators: int = 0
    graph_bytes: int = 0
    context_bytes: int = 0
    arena_used: int = 0
    stack_peak: int | None = None
    load_cycles: int = 0
    context_cycles: int = 0
    compute_cycles: int = 0
    cycles_truncated: bool = False
    output_lengths: list[int] = field(default_factory=list)


class Device:
    """The commands of the runner."""

    def __init__(self, link: Link):
        self.link = link

    def info(self, timeout: float | None = None, attempts: int | None = None) -> Info:
        r = Reader(self.link.request(INFO, timeout=timeout, attempts=attempts))
        numbers = r.unpack("BIIIIIBB")
        strings = [r.string() for _ in range(7)]
        models = [(r.string(), r.unpack("I")) for _ in range(r.unpack("B"))]
        protocol, max_payload, ram, hz, reason, max_operators, crash, failed = numbers
        return Info(protocol, max_payload, ram, hz, reason, max_operators, CRASHES.get(crash, "crash"), failed, *strings, models)

    def write(self, offset: int, data: bytes) -> None:
        """Writes `data` at `offset` of the RAM store, a payload at a time."""
        chunk = MAX_PAYLOAD - 4
        for at in range(0, len(data), chunk):
            self.link.request(WRITE, struct.pack("<I", offset + at) + data[at : at + chunk])

    def crc(self, store: int, at: int, length: int) -> int:
        """The CRC-32 of `length` bytes at `at` of the RAM store, or of the embedded model `at`."""
        return struct.unpack("<I", self.link.request(CRC, struct.pack("<BII", store, at, length)))[0]

    def run(self, store: int, at: int, length: int, arena: int, inputs: list[tuple[int, int]], timeout: float) -> RunResult:
        """Loads the model (`length` bytes at `at` of the RAM store, or the embedded model `at`)
        with the arena at `arena` of the RAM store, and runs it on `inputs`, offsets and lengths
        in the RAM store. Sent once: a run whose answer is lost is not run again behind it."""
        payload = struct.pack("<BIIIB", store, at, length, arena, len(inputs))
        payload += b"".join(struct.pack("<II", *span) for span in inputs)
        r = Reader(self.link.request(RUN, payload, timeout=timeout, attempts=1))
        status = r.unpack("B")
        if status != OK:
            return RunResult(status, r.unpack("B"), r.rest().decode(errors="replace"))
        operators, graph, context, used, stack, load, create, compute, flags, outputs = r.unpack("IIIIIIIIBB")
        return RunResult(
            status,
            operators=operators,
            graph_bytes=graph,
            context_bytes=context,
            arena_used=used,
            stack_peak=None if stack == STACK_UNKNOWN else stack,
            load_cycles=load,
            context_cycles=create,
            compute_cycles=compute,
            cycles_truncated=bool(flags & CYCLES_TRUNCATED),
            output_lengths=[r.unpack("I") for _ in range(outputs)],
        )

    def operators(self, count: int) -> list[tuple[int, str]]:
        """The cycles and the kernel of each of the `count` operators of the last run."""
        operators: list[tuple[int, str]] = []
        while len(operators) < count:
            r = Reader(self.link.request(OPERATORS, struct.pack("<I", len(operators))))
            first, got = r.unpack("IB")
            if first != len(operators) or got == 0:
                raise ProtocolError(f"operators from {first} ({got} of them), asked from {len(operators)}")
            operators += [(r.unpack("I"), r.string()) for _ in range(got)]
        return operators[:count]

    def output(self, index: int, length: int) -> bytes:
        """The `length` bytes of output `index` of the last run."""
        data = bytearray()
        while len(data) < length:
            part = self.link.request(OUTPUT, struct.pack("<BII", index, len(data), length - len(data)))
            if not part:
                raise ProtocolError(f"output {index} ends at {len(data)} bytes, {length} announced")
            data += part
        return bytes(data)

    def reset(self, to: int) -> None:
        """Resets the board, into the firmware or the bootloader. The port goes away."""
        self.link.request(RESET, bytes([to]), attempts=1)

    def fault(self, fault: int) -> None:
        """Makes the firmware take a fault. The port goes away."""
        self.link.request(FAULT, bytes([fault]), attempts=1)
