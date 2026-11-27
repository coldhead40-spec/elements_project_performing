"""Laban effort vector  ->  soft membership of the four elements (+ temporal persistence).

Prototypes are points in quality space (weight, time, space, flow, repetitivity), each 0..1:
    weight  0 light   .. 1 strong
    time    0 sustained .. 1 sudden
    space   0 indirect  .. 1 direct
    flow    0 free      .. 1 bound       (here: 1 - smoothness; a proxy, not intention)
    repet.  0 irregular .. 1 rhythmic    (Layer-3 descriptor, separates air from water)

DEFAULTS BELOW ARE A STARTING HYPOTHESIS (loosely: air~float, earth~press, fire~slash/punch,
water~glide/wring), NOT ground truth. Calibrate them on real bodies with:
    python main.py --calibrate fire      (do it for air, earth, fire, water)
which writes prototypes.json and overrides these numbers.
"""
import json
import os
import numpy as np

ELEMENTS = ["air", "earth", "fire", "water"]
EFFORT = ["weight", "time", "space", "flow", "repetitivity"]

DEFAULT_PROTOTYPES = {
    #          weight time  space flow  repet
    "air":   [0.15, 0.10, 0.25, 0.10, 0.15],   # light, sustained, indirect, free, irregular  (float)
    "earth": [0.35, 0.10, 0.85, 0.15, 0.10],   # slow, direct, grounded stance, not rhythmic   (press)
    "fire":  [0.65, 0.85, 0.30, 0.85, 0.40],   # strong, sudden, jerky                         (slash / punch)
    "water": [0.20, 0.15, 0.40, 0.10, 0.65],   # smooth, flowing, rhythmic, curved             (wave / wring)
}
DIM_W = np.array([1.0, 1.3, 0.8, 0.7, 0.9])    # flow is the weakest proxy -> counts least


def load_prototypes(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return {k: list(v) for k, v in json.load(f).items()}
    return {k: list(v) for k, v in DEFAULT_PROTOTYPES.items()}


class ElementClassifier:
    def __init__(self, cfg, prototypes=None):
        self.cfg = cfg
        self.proto = np.array([(prototypes or load_prototypes(cfg.prototypes_file))[e] for e in ELEMENTS])
        self.effort = np.zeros(len(EFFORT))
        self.level = np.zeros(4)

    def update(self, feat, dt):
        cfg = self.cfg
        act = feat.get("activity", 0.0)
        target_e = np.array([feat.get(k, 0.0) for k in EFFORT])
        a = 1 - np.exp(-dt / cfg.effort_smooth_s)
        self.effort += a * (target_e - self.effort)

        d2 = ((self.effort - self.proto) ** 2 * DIM_W).sum(axis=1)
        logits = -d2 / cfg.softmax_temp
        p = np.exp(logits - logits.max()); p /= p.sum()
        target = p * act * cfg.out_gain

        up = 1 - np.exp(-dt / cfg.attack_s)
        down = 1 - np.exp(-dt / cfg.release_s)
        k = np.where(target > self.level, up, down)
        self.level += k * (target - self.level)
        lvl = np.clip(self.level, 0.0, 1.0)
        out = {e: float(lvl[i]) for i, e in enumerate(ELEMENTS)}
        out["dominant"] = float(np.argmax(lvl)) if lvl.max() > 0.05 else -1.0   # 0 air 1 earth 2 fire 3 water
        return out
