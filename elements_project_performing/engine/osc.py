"""Minimal OSC-over-UDP sender (floats only, one bundle per frame).

Why hand-rolled: zero dependencies, and one *bundle* per frame means TouchDesigner's
OSC In CHOP receives all channels in the same sample (no half-updated frames).
Every value is sent on its own flat address (/air, /weight, ...) so the CHOP channel
names are simply 'air', 'weight', ...
"""
import socket
import struct


def _pad(b: bytes) -> bytes:
    return b + b"\x00" * (4 - len(b) % 4)


def _message(address: str, value: float) -> bytes:
    return _pad(address.encode()) + _pad(b",f") + struct.pack(">f", float(value))


class OscSender:
    def __init__(self, host="127.0.0.1", port=9000):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, values: dict):
        body = b""
        for k, v in values.items():
            m = _message("/" + k, v)
            body += struct.pack(">i", len(m)) + m
        packet = _pad(b"#bundle") + b"\x00\x00\x00\x00\x00\x00\x00\x01" + body
        self.sock.sendto(packet, self.addr)


def decode_bundle(packet: bytes) -> dict:
    """Tiny decoder used by the tests."""
    out, i = {}, 16
    while i < len(packet):
        n = struct.unpack(">i", packet[i:i + 4])[0]
        m = packet[i + 4:i + 4 + n]
        end = m.index(b"\x00")
        addr = m[:end].decode()
        out[addr.lstrip("/")] = struct.unpack(">f", m[-4:])[0]
        i += 4 + n
    return out
