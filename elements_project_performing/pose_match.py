"""Pose matching for the four-element installation.  Pure numpy: no camera, no MediaPipe needed here.

Input : MediaPipe landmarks (33, 4) = x, y (0..1 image coords), z, visibility
Output: one score 0..1 per element pose (water, earth, air, fire)

How it works
  1. landmarks -> a handful of body-size-independent descriptors (joint angles, arm elevation,
     torso lean, stance width ...).  Distances are divided by torso length / shoulder width,
     so it does not matter how far the visitor stands from the camera.
  2. every pose is a template  {descriptor: (target, tolerance)}.
  3. score = exp(-0.5 * cost), cost mixes the mean and the worst normalised error, so ONE clearly
     wrong limb is enough to fail the pose.
  4. each pose is also tested left<->right mirrored (visitors mirror the avatar, so which arm they
     use must not matter).

The default templates are plausible yoga geometry, NOT ground truth.  Calibrate them on real bodies:
press 1-4 in main_poses.py while standing in the pose (writes poses.json, which overrides defaults).
"""
import json
import math
import os

import numpy as np

NAMES = ["water", "earth", "air", "fire"]                # order of the storyboard

# MediaPipe Pose landmark indices
NOSE, LS, RS, LE, RE, LW, RW = 0, 11, 12, 13, 14, 15, 16
LH, RH, LK, RK, LA, RA = 23, 24, 25, 26, 27, 28
VIS = 0.5                                                # minimum landmark visibility

# descriptor -> default tolerance (one "sigma").  Units: degrees, or torso lengths / shoulder widths.
TOL = {"elbow_l": 25, "elbow_r": 25, "arm_l": 22, "arm_r": 22, "knee_min": 22, "knee_max": 15,
       "tilt": 10, "hands_gap": 0.5, "hands_up": 0.3, "stance": 0.5, "foot_lift": 0.3}
MIRROR = {"elbow_l": "elbow_r", "elbow_r": "elbow_l", "arm_l": "arm_r", "arm_r": "arm_l"}


def _t(**kw):
    """template helper: _t(arm_l=90) -> {'arm_l': (90, default tol)}; _t(arm_l=(90, 10)) overrides tol."""
    return {k: (v if isinstance(v, tuple) else (v, TOL[k])) for k, v in kw.items()}


# ---- default templates: read off the four avatar videos (frames at the hold phase) ---------------
# They are a first estimate.  Better: `python poses_from_videos.py` (learns them from the mp4s with
# MediaPipe) and/or press 1-4 in main_poses.py while copying each avatar (see README_poses.md).
DEFAULT_POSES = {
    # Water - arms stretched out to the sides, loose and flowing (slightly below/at shoulder height,
    #         elbows soft), legs together
    "water": _t(arm_l=(100, 28), arm_r=(100, 28), elbow_l=(150, 35), elbow_r=(150, 35),
                hands_gap=(3.6, 0.8), hands_up=(-0.1, 0.3), knee_max=172, knee_min=172, tilt=0),
    # Earth - mountain pose: standing tall, feet together, arms hanging at the sides
    "earth": _t(arm_l=(172, 12), arm_r=(172, 12), elbow_l=(172, 15), elbow_r=(172, 15), hands_up=(-1.0, 0.25),
                knee_max=175, knee_min=175, stance=(0.6, 0.35), foot_lift=(0.0, 0.12), tilt=0),
    # Air - tree pose, hands on hips (elbows out), one foot against the standing leg
    "air": _t(arm_l=(135, 22), arm_r=(135, 22), elbow_l=(95, 30), elbow_r=(95, 30), hands_up=(-0.7, 0.3),
              foot_lift=(0.95, 0.3), knee_max=(175, 12), knee_min=(55, 35), tilt=0),
    # Fire - both arms overhead, elbows out, hands joined above the head (diamond / prayer overhead)
    "fire": _t(arm_l=(55, 25), arm_r=(55, 25), elbow_l=(70, 35), elbow_r=(70, 35), hands_gap=(0.4, 0.6),
               hands_up=(0.9, 0.3), knee_max=172, knee_min=172, tilt=0),
}


# ---- optional pyEyesWeb primitive ---------------------------------------------------------------
class Primitives:
    """Uses pyEyesWeb's Equilibrium (barycentre between the two feet) when the installed version
    answers like the documented one; otherwise a small numpy equivalent.  `backend` tells which."""

    def __init__(self):
        self.backend, self._eq = "numpy-fallback", None
        try:
            from pyeyesweb.low_level import Equilibrium
            eq = Equilibrium(left_foot_idx=0, right_foot_idx=1, barycenter_idx=2)
            v = self._value(eq.compute(np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])))
            if v is not None and 0.9 <= v <= 1.0001:              # centred barycentre must score ~1
                self._eq, self.backend = eq, "pyeyesweb"
        except Exception:
            pass

    @staticmethod
    def _value(r):
        if isinstance(r, dict):
            r = r.get("value", r.get("equilibrium"))
            return None if r is None else float(r)
        v = getattr(r, "value", None)
        return None if v is None else float(v)

    def balance(self, left, right, bary, torso):
        if self._eq is not None:
            try:
                v = self._value(self._eq.compute(np.array([left, right, bary], dtype=float)))
                if v is not None:
                    return float(np.clip(v, 0, 1))
            except Exception:
                self._eq, self.backend = None, "numpy-fallback"
        half = max(abs(left[0] - right[0]) / 2, 0.15 * torso)
        return float(np.clip(1 - abs(bary[0] - (left[0] + right[0]) / 2) / (2 * half), 0, 1))


# ---- geometry -----------------------------------------------------------------------------------
def _angle(a, b, c):
    """angle at b in degrees (180 = straight)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float("nan") if n < 1e-9 else math.degrees(math.acos(np.clip(np.dot(v1, v2) / n, -1, 1)))


def _hull_area(pts):
    p = sorted(map(tuple, pts))
    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h
    hull = half(p)[:-1] + half(p[::-1])[:-1]
    x, y = np.array(hull).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class PoseMatcher:
    def __init__(self, aspect=16 / 9, poses_path=None, tol_scale=1.0):
        self.aspect, self.tol_scale = aspect, tol_scale
        self.poses_path = poses_path
        self.poses = {k: dict(v) for k, v in DEFAULT_POSES.items()}
        self.custom = []
        if poses_path and os.path.exists(poses_path):
            with open(poses_path) as f:
                for name, tpl in json.load(f).items():
                    if name in self.poses:
                        self.poses[name] = {k: (float(v[0]), float(v[1])) for k, v in tpl.items()}
                        self.custom.append(name)
        self.prim = Primitives()
        self._prev, self._speed = None, 0.0

    # -- descriptors ----------------------------------------------------------------------------
    def analyse(self, lm, t):
        """-> (features dict | None, info dict).  info: present, full_body, steady, balance."""
        info = dict(present=0.0, full_body=0.0, steady=0.0, balance=0.0)
        if lm is None or min(lm[[LS, RS, LH, RH], 3]) < VIS:
            self._prev = None
            return None, info
        P = lm[:, :2] * np.array([self.aspect, 1.0])             # isotropic image coordinates
        ok = lm[:, 3] >= VIS
        info["present"] = 1.0
        legs = [LK, RK, LA, RA]
        info["full_body"] = float(all(ok[legs]) and all((lm[[LA, RA], 1] < 1.0) & (lm[[LA, RA], 1] > 0.0)))

        ms, mh = (P[LS] + P[RS]) / 2, (P[LH] + P[RH]) / 2
        up = ms - mh
        torso = np.linalg.norm(up) + 1e-9
        sw = np.linalg.norm(P[LS] - P[RS]) + 1e-9
        f = {}

        def have(*idx):
            return all(ok[i] for i in idx)

        for side, (s, e, w) in (("l", (LS, LE, LW)), ("r", (RS, RE, RW))):
            f["elbow_" + side] = _angle(P[s], P[e], P[w]) if have(s, e, w) else float("nan")
        # arm elevation = angle between "torso up" and upper arm (0 = overhead, 90 = out, 180 = down)
        for side, (s, e) in (("l", (LS, LE)), ("r", (RS, RE))):
            v = P[e] - P[s]
            f["arm_" + side] = (math.degrees(math.acos(np.clip(np.dot(up, v) / (torso * (np.linalg.norm(v) + 1e-9)), -1, 1)))
                                if have(e) else float("nan"))
        f["tilt"] = math.degrees(math.atan2(up[0], -up[1]))
        f["hands_gap"] = np.linalg.norm(P[LW] - P[RW]) / sw if have(LW, RW) else float("nan")
        f["hands_up"] = (ms[1] - (P[LW][1] + P[RW][1]) / 2) / torso if have(LW, RW) else float("nan")
        if have(*legs):
            kl, kr = _angle(P[LH], P[LK], P[LA]), _angle(P[RH], P[RK], P[RA])
            f["knee_min"], f["knee_max"] = min(kl, kr), max(kl, kr)
            f["stance"] = abs(P[LA][0] - P[RA][0]) / sw
            f["foot_lift"] = abs(P[LA][1] - P[RA][1]) / torso
            info["balance"] = self.prim.balance(P[LA], P[RA], mh, torso)
        else:
            for k in ("knee_min", "knee_max", "stance", "foot_lift"):
                f[k] = float("nan")

        # steadiness of the hold: joints are low-pass filtered first (camera jitter is not movement), then the
        # median joint speed in torso lengths / s is turned into 0..1
        joints = P[[LS, RS, LE, RE, LW, RW, LH, RH]]
        if self._prev is None:
            sm, dtp = joints.copy(), None
        else:
            sm = self._prev[1] + (joints - self._prev[1]) * 0.12
            dtp = t - self._prev[0]
            if dtp > 0:
                sp = np.median(np.linalg.norm(sm - self._prev[1], axis=1)) / torso / dtp
                self._speed += (sp - self._speed) * 0.2
        self._prev = (t, sm)
        info["steady"] = float(math.exp(-self._speed / 0.25))
        return f, info

    # -- matching -------------------------------------------------------------------------------
    @staticmethod
    def _mirrored(f):
        g = {MIRROR.get(k, k): v for k, v in f.items()}
        g["tilt"] = -f["tilt"]
        return g

    def _score(self, f, tpl):
        z2 = []
        for k, (target, tol) in tpl.items():
            v = f.get(k, float("nan"))
            z = 3.0 if v != v else abs(v - target) / (tol * self.tol_scale)   # missing -> penalised
            z2.append(min(z, 4.0) ** 2)
        cost = 0.5 * float(np.mean(z2)) + 0.25 * float(np.max(z2))
        return math.exp(-0.5 * cost)

    def scores(self, f):
        if f is None:
            return {n: 0.0 for n in NAMES}
        fm = self._mirrored(f)
        return {n: max(self._score(f, self.poses[n]), self._score(fm, self.poses[n])) for n in NAMES}

    # -- calibration ----------------------------------------------------------------------------
    def record(self, name, feats):
        """feats: list of descriptor dicts collected while the person holds the pose."""
        keys = [k for k in TOL if all(k in x for x in feats)]
        tpl = {}
        for k in keys:
            vals = np.array([x[k] for x in feats], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) >= max(5, len(feats) // 2):
                tpl[k] = (float(np.median(vals)), float(TOL[k]))
        self.poses[name] = tpl
        if name not in self.custom:
            self.custom.append(name)
        if self.poses_path:
            data = {}
            if os.path.exists(self.poses_path):
                with open(self.poses_path) as fh:
                    data = json.load(fh)
            data[name] = {k: list(v) for k, v in tpl.items()}
            with open(self.poses_path, "w") as fh:
                json.dump(data, fh, indent=1)
        return tpl


# ---- synthetic skeletons (for tests and for running without a camera) ---------------------------
_T, _SW, _LU, _LF, _LT, _LS = 0.22, 0.176, 0.132, 0.121, 0.187, 0.187      # body proportions (image height = 1)
# per pose: tilt (deg), arms {side: (upper, fore)}, legs {side: (thigh, shin)}; degrees from "straight down",
# positive = away from the body midline.  side -1 = image left, +1 = image right.
_SPECS = {
    "water":   dict(tilt=0, arms={-1: (100, 110), 1: (100, 80)}, legs={-1: (3, 0), 1: (3, 0)}),
    "earth":   dict(tilt=0, arms={-1: (6, 2), 1: (6, 2)}, legs={-1: (3, 0), 1: (3, 0)}),
    "air":     dict(tilt=0, arms={-1: (42, -55), 1: (42, -55)}, legs={-1: (45, -75), 1: (0, 0)}),
    "fire":    dict(tilt=0, arms={-1: (118, 250), 1: (118, 250)}, legs={-1: (3, 0), 1: (3, 0)}),
    "neutral": dict(tilt=0, arms={-1: (40, 30), 1: (40, 30)}, legs={-1: (3, 0), 1: (3, 0)}),   # hands half-raised
}


def synthetic_skeleton(name, aspect=16 / 9, mirror=False, jitter=0.0, rng=None):
    """Plausible 33x4 landmark array of a person doing pose `name` (or 'neutral')."""
    spec = _SPECS[name]
    rad = math.radians
    hip = np.array([aspect / 2, 0.0])
    P = {LH: hip + [-0.07, 0], RH: hip + [0.07, 0]}
    for side, (h, k, a) in ((-1, (LH, LK, LA)), (1, (RH, RK, RA))):
        tu, ts = spec["legs"][side]
        P[k] = P[h] + _LT * np.array([side * math.sin(rad(tu)), math.cos(rad(tu))])
        P[a] = P[k] + _LS * np.array([side * math.sin(rad(ts)), math.cos(rad(ts))])
    upper = {NOSE: hip + [0, -_T - 0.07]}
    for side, (s, e, w) in ((-1, (LS, LE, LW)), (1, (RS, RE, RW))):
        pu, pf = spec["arms"][side]
        upper[s] = hip + [side * _SW / 2, -_T]
        upper[e] = upper[s] + _LU * np.array([side * math.sin(rad(pu)), math.cos(rad(pu))])
        upper[w] = upper[e] + _LF * np.array([side * math.sin(rad(pf)), math.cos(rad(pf))])
    th = rad(spec["tilt"])
    for i, p in upper.items():                      # lean the upper body about the hip centre
        r = p - hip
        P[i] = hip + np.array([r[0] * math.cos(th) - r[1] * math.sin(th), r[0] * math.sin(th) + r[1] * math.cos(th)])
    dy = 0.92 - max(P[LA][1], P[RA][1])
    lm = np.zeros((33, 4))
    lm[:, 3] = 1.0
    for i, p in P.items():
        lm[i, :2] = [p[0] / aspect, p[1] + dy]
    for i in range(33):                             # fill unused landmarks with the torso centre
        if i not in P:
            lm[i, :2] = [0.5, 0.5]
    if mirror:
        lm[:, 0] = 1 - lm[:, 0]
        for a, b in ((LS, RS), (LE, RE), (LW, RW), (LH, RH), (LK, RK), (LA, RA)):
            lm[[a, b]] = lm[[b, a]]
    if jitter:
        rng = rng or np.random.default_rng(0)
        lm[:, :2] += rng.normal(0, jitter, (33, 2))
    return lm
