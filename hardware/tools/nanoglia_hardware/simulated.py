"""A simulated device: the runner of hardware/runner rewritten in a few lines of Python, behind
the transport interface, for the tests of the host tool and its dry run.

It speaks the protocol but runs no model: an oracle gives the outputs of a run, in the dry run the
expected outputs of the golden case. A dry run therefore checks the host tool (the layout of the
RAM store, the transfers, the comparison, the results file, the recovery), never nanoglia.

`Faults` makes it misbehave as a board can: corrupt or lose an answer, stop answering during a run
and restart (as the watchdog of the firmware does), come back in the bootloader, or never come
back.
"""

import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable

from . import protocol as p

# What an oracle returns for a model and its inputs: the outputs and the kernel of each operator,
# or the error code of nanoglia and its message.
Oracle = Callable[[bytes, list[bytes]], tuple[list[bytes], list[str]] | tuple[int, str]]


@dataclass
class Faults:
    """What goes wrong, by the number of the RUN (from 1) or of the answer (from 1)."""

    # Answers with a flipped bit, and answers never sent.
    corrupt_answers: set[int] = field(default_factory=set)
    lost_answers: set[int] = field(default_factory=set)
    # Runs during which the firmware hangs: no answer, then a restart by its watchdog.
    hang_runs: set[int] = field(default_factory=set)
    # Runs during which the firmware panics.
    panic_runs: set[int] = field(default_factory=set)
    # After a hang or a panic, the board comes back in the bootloader (as after several crashes).
    back_in_bootloader: bool = False
    # After a hang or a panic, the board never comes back.
    dead: bool = False


class SimulatedDevice:
    """The device and its transport in one."""

    RAM_BYTES = 229 * 1024
    # The device runs inside `write` and `read`: an empty read means nothing will come.
    instant = True

    def __init__(self, oracle: Oracle, models: list[tuple[str, bytes]] | None = None, faults: Faults | None = None):
        self.oracle = oracle
        self.models = models or []
        self.faults = faults or Faults()
        self.ram = bytearray(self.RAM_BYTES)
        self.incoming = bytearray()
        self.outgoing = bytearray()
        self.results: tuple[list[bytes], list[str]] | None = None
        self.runs = 0
        self.answers = 0
        self.resets: list[int] = []
        # "running", "crashed" (will restart), "bootloader" or "dead".
        self.state = "running"
        self.crash = 0
        self.crash_message = ""
        self.failed_boots = 0

    # --- Transport.

    def write(self, data: bytes) -> None:
        if self.state != "running":
            # A port that is gone refuses the write; a firmware that hangs swallows it.
            if self.state in ("dead", "bootloader"):
                raise p.Disconnected(f"simulated device: {self.state}")
            return
        self.incoming += data
        while self._frame():
            pass

    def read(self, size: int, timeout: float) -> bytes:
        data = bytes(self.outgoing[:size])
        del self.outgoing[:size]
        return data

    def close(self) -> None:
        pass

    # --- What a board does behind the back of the protocol.

    def restart(self) -> None:
        """The restart after a crash: the firmware answers again and reports the crash."""
        if self.state != "crashed":
            return
        self.incoming.clear()
        self.outgoing.clear()
        self.results = None
        if self.faults.dead:
            self.state = "dead"
        elif self.faults.back_in_bootloader:
            self.state = "bootloader"
        else:
            self.state = "running"
            self.failed_boots = 1

    def flashed(self, models: list[tuple[str, bytes]]) -> None:
        """A new firmware was flashed, with these models linked in, and started."""
        self.models = models
        self.state = "running"
        self.crash, self.crash_message, self.failed_boots = 0, "", 0
        self.incoming.clear()
        self.outgoing.clear()
        self.results = None

    # --- The runner.

    def _send(self, kind: int, seq: int, payload: bytes = b"") -> None:
        self.answers += 1
        if self.answers in self.faults.lost_answers:
            return
        frame = bytearray(p.encode(kind, seq, payload))
        if self.answers in self.faults.corrupt_answers:
            frame[len(frame) // 2] ^= 0x10
        self.outgoing += frame

    def _error(self, seq: int, code: int, message: str) -> None:
        self._send(p.ERROR, seq, bytes([code]) + message.encode())

    def _frame(self) -> bool:
        """Handles one frame of `incoming`; False when there is none left."""
        start = self.incoming.find(p.SOF)
        if start < 0:
            self.incoming.clear()
            return False
        del self.incoming[:start]
        if len(self.incoming) < p.HEADER:
            return False
        version, kind, seq, length = struct.unpack_from("<BBBH", self.incoming, 2)
        if length > p.MAX_PAYLOAD:
            del self.incoming[:2]
            self._error(seq, p.BAD_FRAME, "BadFrame")
            return True
        end = p.HEADER + length + p.TRAILER
        if len(self.incoming) < end:
            return False
        body = bytes(self.incoming[2 : p.HEADER + length])
        (crc,) = struct.unpack_from("<I", self.incoming, p.HEADER + length)
        del self.incoming[:end]
        if crc != zlib.crc32(body):
            self._error(seq, p.BAD_CRC, "BadCrc")
        elif version != p.VERSION:
            self._error(seq, p.BAD_VERSION, "Version")
        else:
            try:
                self._command(kind, seq, body[5:])
            except (struct.error, IndexError):
                self._error(seq, p.BAD_ARGUMENT, "malformed request")
        return self.state == "running"

    def _span(self, offset: int, length: int, limit: int | None = None) -> bytes | None:
        limit = len(self.ram) if limit is None else limit
        return bytes(self.ram[offset : offset + length]) if offset + length <= limit else None

    def _command(self, kind: int, seq: int, payload: bytes) -> None:
        if kind not in (p.OPERATORS, p.OUTPUT):
            results, self.results = self.results, None
        else:
            results = self.results
        if kind == p.INFO:
            numbers = struct.pack("<BIIIIIBB", p.VERSION, p.MAX_PAYLOAD, len(self.ram), 0, 0, 256, self.crash, self.failed_boots)
            strings = ["simulated 0.0.0", "none", "none", "no kernel: the outputs are the expected ones", "simulated device", "none", self.crash_message]
            answer = numbers + b"".join(bytes([len(s)]) + s.encode() for s in strings) + bytes([len(self.models)])
            for name, data in self.models:
                answer += bytes([len(name)]) + name.encode() + struct.pack("<I", len(data))
            self._send(kind | p.RESPONSE, seq, answer)
        elif kind == p.WRITE:
            (offset,) = struct.unpack_from("<I", payload)
            data = payload[4:]
            if offset + len(data) > len(self.ram):
                return self._error(seq, p.BAD_ARGUMENT, "outside the RAM store")
            self.ram[offset : offset + len(data)] = data
            self._send(kind | p.RESPONSE, seq)
        elif kind == p.CRC:
            store, at, length = struct.unpack("<BII", payload)
            data = self._model(store, at, length, len(self.ram))
            if data is None:
                return self._error(seq, p.BAD_ARGUMENT, "outside the RAM store, or no such embedded model")
            self._send(kind | p.RESPONSE, seq, struct.pack("<I", zlib.crc32(data)))
        elif kind == p.RUN:
            self._run(seq, payload)
        elif kind == p.OPERATORS:
            if results is None:
                return self._error(seq, p.NO_RUN, "no results: RUN first")
            (first,) = struct.unpack("<I", payload)
            kernels = results[1][first : first + 20]
            body = b"".join(struct.pack("<IB", 1000 + first + i, len(k)) + k.encode() for i, k in enumerate(kernels))
            self._send(kind | p.RESPONSE, seq, struct.pack("<IB", first, len(kernels)) + body)
        elif kind == p.OUTPUT:
            if results is None:
                return self._error(seq, p.NO_RUN, "no results: RUN first")
            index, offset, length = struct.unpack("<BII", payload)
            if index >= len(results[0]) or offset > len(results[0][index]):
                return self._error(seq, p.BAD_ARGUMENT, "no such output, or an offset past its end")
            self._send(kind | p.RESPONSE, seq, results[0][index][offset : offset + min(length, p.MAX_PAYLOAD)])
        elif kind == p.RESET:
            if payload not in (b"\x00", b"\x01"):
                return self._error(seq, p.BAD_ARGUMENT, "malformed RESET")
            self._send(kind | p.RESPONSE, seq)
            self.resets.append(payload[0])
            self.state = "bootloader" if payload[0] == p.BOOTLOADER else "running"
            self.crash, self.crash_message = 0, ""
        elif kind == p.FAULT:
            if payload not in (b"\x01", b"\x02", b"\x03"):
                return self._error(seq, p.BAD_ARGUMENT, "malformed FAULT")
            self._send(kind | p.RESPONSE, seq)
            self._crash(payload[0], f"{p.CRASHES[payload[0]]} requested by the host")
        else:
            self._error(seq, p.UNKNOWN_COMMAND, f"unknown command {kind}")

    def _model(self, store: int, at: int, length: int, limit: int) -> bytes | None:
        if store == p.RAM:
            return self._span(at, length, limit)
        if store == p.EMBEDDED and at < len(self.models) and len(self.models[at][1]) == length:
            return self.models[at][1]
        return None

    def _crash(self, kind: int, message: str) -> None:
        self.state = "crashed"
        self.crash, self.crash_message = kind, message

    def _run(self, seq: int, payload: bytes) -> None:
        store, at, length, arena, count = struct.unpack_from("<BIIIB", payload)
        spans = [struct.unpack_from("<II", payload, 14 + 8 * i) for i in range(count)]
        if len(payload) != 14 + 8 * count or arena > len(self.ram):
            return self._error(seq, p.BAD_ARGUMENT, "malformed RUN")
        model = self._model(store, at, length, arena)
        inputs = [self._span(offset, size, arena) for offset, size in spans]
        if model is None or any(data is None for data in inputs):
            return self._error(seq, p.BAD_ARGUMENT, "the model or an input is outside the RAM store or in the arena")
        self.runs += 1
        if self.runs in self.faults.hang_runs:
            return self._crash(3, "watchdog: nothing moved for 20 s")
        if self.runs in self.faults.panic_runs:
            return self._crash(1, "panicked at kernel.rs:1:1: simulated")
        outcome = self.oracle(model, inputs)
        if isinstance(outcome[0], int):
            code, message = outcome
            return self._send(p.RUN | p.RESPONSE, seq, bytes([p.LOAD, code]) + message.encode())
        outputs, kernels = outcome
        self.results = (outputs, kernels)
        answer = struct.pack("<BIIIIIIIIBB", p.OK, len(kernels), 4096, 8192, 12288, 2048, 100, 10, 1000 * len(kernels), 0, len(outputs))
        answer += b"".join(struct.pack("<I", len(output)) for output in outputs)
        self._send(p.RUN | p.RESPONSE, seq, answer)
