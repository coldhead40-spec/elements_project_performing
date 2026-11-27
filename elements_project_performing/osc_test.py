"""Fake pose detector: no camera needed. Run it, watch oscin1 in TouchDesigner, stop with Ctrl+C."""
import socket
import struct
import time

HOST, PORT = '127.0.0.1', 9000   # same computer as TouchDesigner


def pad(b):
    return b + b'\0' * (4 - len(b) % 4)


def message(address, value):
    return pad(address.encode()) + pad(b',f') + struct.pack('>f', value)


names = ['present', 'full_body', 'steady', 'match_water', 'match_earth', 'match_air', 'match_fire']
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print('sending to', HOST, PORT, '- Ctrl+C to stop')
while True:
    for n in names:
        sock.sendto(message('/' + n, 1.0), (HOST, PORT))
    time.sleep(1 / 30)
