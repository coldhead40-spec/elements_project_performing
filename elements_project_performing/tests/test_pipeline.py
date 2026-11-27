"""Run:  python tests/test_pipeline.py      (no camera, no TouchDesigner, no pyeyesweb needed)

These tests check that the CODE is wired correctly and that the classifier separates four
synthetic movement styles. They do NOT validate the Laban mapping on real people -- do that
with `python main.py --calibrate <element>`.
"""
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.config import Config
from engine.elements import ELEMENTS, ElementClassifier
from engine.features import MotionAnalyzer
from engine.osc import OscSender, decode_bundle
from engine.pose_source import SimPose
from engine.smoothness import sparc_numpy


def run_mode(mode, jitter=0.0015, seconds=45):
    cfg = Config()
    src = SimPose(mode, jitter=jitter)
    still = SimPose("still", jitter=jitter)
    an, cls = MotionAnalyzer(cfg, aspect=src.aspect), ElementClassifier(cfg)
    rows = []
    for i in range(int(seconds * 30)):
        t, lm, _ = (still if (mode != "still" and i / 30 < 8) else src).frame(i)   # 8 s of stillness first
        f = an.update(t, lm)
        el = cls.update(f, 1 / 30)
        if t > 30:
            rows.append([f.get("activity", 0)] + [el[e] for e in ELEMENTS])
    return np.mean(rows, axis=0)


def test_sparc_orders_smoothness():
    t = np.linspace(0, 1, 60)
    smooth = np.sin(np.pi * t) ** 2
    jerky = smooth + np.random.default_rng(0).normal(0, 0.15, 60).clip(-0.3, 0.3)
    assert sparc_numpy(smooth, 30) > sparc_numpy(np.abs(jerky), 30)


def test_stillness_is_silent():
    r = run_mode("still")
    assert r[0] < 0.1 and r[1:].max() < 0.1, r


def test_each_element_wins_its_own_movement():
    for jitter in (0.0007, 0.0015, 0.003):
        for mode in ELEMENTS:
            r = run_mode(mode, jitter)
            assert ELEMENTS[int(np.argmax(r[1:]))] == mode, (mode, jitter, r.round(2))


def test_osc_roundtrip():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0)); rx.settimeout(2)
    OscSender("127.0.0.1", rx.getsockname()[1]).send({"air": 0.25, "fire": 0.75, "rw_x": 0.5})
    got = decode_bundle(rx.recv(2048))
    assert abs(got["air"] - 0.25) < 1e-6 and abs(got["fire"] - 0.75) < 1e-6 and abs(got["rw_x"] - 0.5) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
