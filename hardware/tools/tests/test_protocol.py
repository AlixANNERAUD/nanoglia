"""The link of the host: frames, retries, noise, errors."""

import struct
import zlib

import pytest

from nanoglia_hardware import protocol as p


class Scripted:
    """A transport that answers each write with the next entry of a script."""

    def __init__(self, script):
        self.script = list(script)
        self.written = []
        self.outgoing = bytearray()

    def write(self, data):
        self.written.append(bytes(data))
        if self.script:
            answer = self.script.pop(0)
            self.outgoing += answer(data) if callable(answer) else answer

    def read(self, size, timeout):
        data = bytes(self.outgoing[:size])
        del self.outgoing[:size]
        return data

    def close(self):
        pass


def answer(payload=b"", kind=None, seq=None):
    """The response of a device to the request it receives."""

    def build(request):
        return p.encode(request[3] | p.RESPONSE if kind is None else kind, request[4] if seq is None else seq, payload)

    return build


def link(script, **options):
    transport = Scripted(script)
    return p.Link(transport, timeout=0.02, **options), transport


def test_encode_matches_the_firmware():
    # The frame of `seal_writes_header_and_crc` in runner/src/frame.rs.
    frame = p.encode(0x81, 7, b"abc")
    assert frame[:7] == b"NG" + bytes([p.VERSION, 0x81, 7, 3, 0])
    assert frame[7:10] == b"abc"
    assert struct.unpack("<I", frame[10:])[0] == zlib.crc32(frame[2:10])
    with pytest.raises(ValueError):
        p.encode(1, 0, bytes(p.MAX_PAYLOAD + 1))


def test_a_request_returns_the_payload_of_its_response():
    lnk, transport = link([answer(b"hello")])
    assert lnk.request(p.CRC, b"x") == b"hello"
    assert transport.written == [p.encode(p.CRC, 1, b"x")]
    assert (lnk.retries, lnk.dropped) == (0, 0)


def test_noise_and_stale_frames_are_dropped():
    stale = p.encode(p.CRC | p.RESPONSE, 200, b"old")
    lnk, _ = link([lambda request: b"AT\r\nNNG" + stale + b"N" + answer(b"new")(request)])
    assert lnk.request(p.CRC) == b"new"
    assert lnk.dropped >= 1


def test_a_corrupted_answer_is_asked_again():
    def corrupted(request):
        frame = bytearray(answer(b"payload")(request))
        frame[9] ^= 0x40
        return bytes(frame)

    lnk, transport = link([corrupted, answer(b"payload")])
    assert lnk.request(p.CRC) == b"payload"
    assert len(transport.written) == 2 and transport.written[0] == transport.written[1]
    assert lnk.retries == 1


def test_a_truncated_answer_is_asked_again():
    lnk, _ = link([lambda request: answer(b"payload")(request)[:9], answer(b"payload")])
    assert lnk.request(p.CRC) == b"payload"
    assert lnk.retries == 1


def test_a_frame_the_device_received_broken_is_sent_again():
    broken = answer(bytes([p.BAD_CRC]) + b"BadCrc", kind=p.ERROR)
    lnk, transport = link([broken, answer(b"ok")])
    assert lnk.request(p.WRITE, b"data") == b"ok"
    assert len(transport.written) == 2


def test_silence_ends_in_no_answer():
    lnk, transport = link([])
    with pytest.raises(p.NoAnswer, match="3 attempt"):
        lnk.request(p.INFO)
    assert len(transport.written) == 3
    lnk, transport = link([])
    with pytest.raises(p.NoAnswer):
        lnk.request(p.RESET, b"\x00", attempts=1)
    assert len(transport.written) == 1


def test_a_refused_request_is_not_sent_again():
    refused = answer(bytes([p.BAD_ARGUMENT]) + b"outside the RAM store", kind=p.ERROR)
    lnk, transport = link([refused, answer(b"never")])
    with pytest.raises(p.DeviceError, match="outside the RAM store") as error:
        lnk.request(p.WRITE)
    assert error.value.code == p.BAD_ARGUMENT
    assert len(transport.written) == 1


def test_other_versions_and_kinds_are_protocol_errors():
    def other_version(request):
        body = bytes([p.VERSION + 1, request[3] | p.RESPONSE, request[4], 0, 0])
        return p.SOF + body + struct.pack("<I", zlib.crc32(body))

    lnk, _ = link([other_version])
    with pytest.raises(p.ProtocolError, match="version"):
        lnk.request(p.INFO)
    lnk, _ = link([answer(kind=p.RUN | p.RESPONSE)])
    with pytest.raises(p.ProtocolError):
        lnk.request(p.INFO)


def test_an_oversized_length_is_noise():
    lnk, _ = link([lambda request: b"NG" + bytes([p.VERSION, 0x81, request[4], 0xFF, 0xFF]) + answer(b"ok")(request)])
    assert lnk.request(p.INFO) == b"ok"


def test_device_reads_what_does_not_fit_in_a_frame():
    output = bytes(range(256)) * 10

    def serve(request):
        index, offset, length = struct.unpack("<BII", request[7:-4])
        return answer(output[offset : offset + min(length, p.MAX_PAYLOAD)])(request)

    def operators(request):
        (first,) = struct.unpack("<I", request[7:-4])
        names = [f"rust/kernel_{i}" for i in range(first, min(first + 3, 8))]
        body = b"".join(struct.pack("<IB", 10 * (first + i), len(n)) + n.encode() for i, n in enumerate(names))
        return answer(struct.pack("<IB", first, len(names)) + body)(request)

    lnk, transport = link([serve] * 3 + [operators] * 3)
    device = p.Device(lnk)
    assert device.output(0, len(output)) == output
    assert device.operators(8) == [(10 * i, f"rust/kernel_{i}") for i in range(8)]
    # An output that ends early is an error, not a loop.
    lnk, _ = link([answer(b"")])
    with pytest.raises(p.ProtocolError):
        p.Device(lnk).output(0, 4)
