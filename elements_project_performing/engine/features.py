"""Pose landmarks  ->  the four layers of movement qualities from the project brief.

Layer 1  position / trajectory     : hands, centre of mass, kinesphere extent, level
Layer 2  velocity / acceleration   : speeds, quantity of motion (QoM)
Layer 3  fluidity / impulsivity / repetitivity
Layer 4  (shown by the installation) -> Laban Effort factors, built from layers 1-3
         (repetitivity is kept as a 5th descriptor: rhythmic vs. non-rhythmic):

    Laban factor   polarity (0 .. 1)        proxy used here
    ------------   ----------------------   -------------------------------------------
    Weight         light .. strong          log RMS acceleration of the leading wrist
                                            (+ 'grounded' stance when legs are visible)
    Time           sustained .. sudden      impulsivity = abruptness of speed change (p95 |d speed/dt| / p95 speed)
    Space          indirect .. direct       straightness of the leading wrist's path
    Flow           free .. bound            1 - fluidity (SPARC of the wrist speed profile)

All distances are in TORSO LENGTHS (shoulder-mid to hip-mid), so results do not depend on how
far the visitor stands from the camera. y is flipped so that +y is UP.
"""
from collections import deque
import numpy as np
from scipy.signal import savgol_filter

from .filters import OneEuro
from .smoothness import SparcBackend

# MediaPipe Pose landmark ids
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
TRACKED = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
_ix = {j: i for i, j in enumerate(TRACKED)}
I_SH = [_ix[L_SH], _ix[R_SH]]
I_HIP = [_ix[L_HIP], _ix[R_HIP]]
I_WR = [_ix[L_WR], _ix[R_WR]]
I_AN = [_ix[L_AN], _ix[R_AN]]
# limbs that "express" count more in quantity of motion than the trunk
JOINT_W = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
JOINT_W = JOINT_W / JOINT_W.sum()


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))


SG_WIN = 7          # Savitzky-Golay window (frames): ~0.23 s at 30 fps. Raw finite differences of
SG_ORD = 3          # webcam landmarks are noise-dominated, especially the 2nd derivative.


def sg(x, deriv, dt):
    """Smoothed derivative along axis 0 (needs len(x) >= SG_WIN)."""
    return savgol_filter(x, SG_WIN, SG_ORD, deriv=deriv, delta=dt, axis=0, mode="interp")


class MotionAnalyzer:
    def __init__(self, cfg, aspect=16 / 9):
        self.cfg, self.aspect = cfg, aspect
        self.filt = OneEuro(cfg.oe_min_cutoff, cfg.oe_beta)
        n = int(cfg.long_window_s * cfg.fps_nominal * 1.6)
        self.T = deque(maxlen=n)          # timestamps
        self.PT = deque(maxlen=n)         # all tracked joints (13, 2) in torso units
        self.Q = deque(maxlen=n)          # quantity of motion (scalar)
        self.torso = None
        self.rate = cfg.fps_nominal
        self.sparc = SparcBackend(cfg.fps_nominal, int(cfg.window_s * cfg.fps_nominal))
        self.sparc_value = float("nan")
        self.backend = self.sparc.name
        self.floor = cfg.qom_noise        # estimate of 'standing still' QoM (= landmark jitter)
        self.t0, self.calib_q = None, []

    def idle(self):
        return {"present": 0.0, "activity": 0.0}

    def reset(self):                      # person lost: keep the learned floor, restart the buffers
        self.filt.reset(); self.T.clear(); self.PT.clear(); self.Q.clear()

    def update(self, t, lm):
        """lm: (33,4) array of x,y,z,visibility (MediaPipe normalised) or None."""
        cfg = self.cfg
        if lm is None:
            self.reset()
            return self.idle()

        raw = np.asarray(lm, float)
        mir = (lambda x: 1.0 - x) if cfg.mirror else (lambda x: x)
        xy = np.stack([mir(raw[:, 0]) * self.aspect, -raw[:, 1]], axis=1)      # y up, aspect-corrected
        vis = raw[:, 3] if raw.shape[1] > 3 else np.ones(len(raw))

        torso = np.linalg.norm(xy[[L_SH, R_SH]].mean(0) - xy[[L_HIP, R_HIP]].mean(0))
        self.torso = torso if self.torso is None else 0.95 * self.torso + 0.05 * torso
        pts = self.filt(xy[TRACKED], t) / max(self.torso, 1e-3)                # (13, 2) torso units

        self.T.append(t); self.PT.append(pts)
        T = np.fromiter(self.T, float)
        if len(T) > 1:
            self.rate = 0.95 * self.rate + 0.05 / max(T[-1] - T[-2], 1e-3)

        out = {"present": 1.0, "activity": 0.0,
               "lw_x": float(mir(raw[L_WR, 0])), "lw_y": float(1 - raw[L_WR, 1]),     # layer 1, screen space
               "rw_x": float(mir(raw[R_WR, 0])), "rw_y": float(1 - raw[R_WR, 1]),
               "com_x": float(mir(raw[[L_HIP, R_HIP], 0].mean())), "com_y": float(1 - raw[[L_HIP, R_HIP], 1].mean())}
        if len(T) < SG_WIN + 8:                                                  # warm-up guard
            return out

        PT = np.array(self.PT)                                                   # (n, 13, 2)
        dt = float(np.median(np.diff(T[-30:])))

        # --- layer 2: quantity of motion from smoothed joint velocities ------------
        k = SG_WIN * 2
        v_all = sg(PT[-k:], 1, dt)[-1]                                           # (13, 2)
        qom = float(JOINT_W @ np.linalg.norm(v_all, axis=1))
        self.Q.append(qom)
        Qa = np.fromiter(self.Q, float)
        qom_s = float(Qa[T[-len(Qa):] >= t - 0.5].mean())
        # Jitter floor. (1) For the first cfg.still_calib_s seconds the visitor/operator stands still
        # and the floor is SET to the measured QoM. (2) Afterwards it is a lower envelope that drops
        # fast and creeps up very slowly (so it cannot swallow gentle gestures, but still re-learns
        # if the lighting changes).
        if self.t0 is None:
            self.t0 = t
        calibrating = cfg.still_calib_s > 0 and t - self.t0 < cfg.still_calib_s
        if calibrating:
            self.calib_q.append(qom_s)
            self.floor = float(np.clip(1.15 * np.mean(self.calib_q), 0.005, cfg.qom_floor_max))
        else:
            tau = 0.5 if qom_s < self.floor else cfg.floor_rise_s
            self.floor = float(np.clip(self.floor + (qom_s - self.floor) * min(dt / tau, 1.0),
                                       0.005, cfg.qom_floor_max))
        activity = 0.0 if calibrating else smoothstep(qom_s - self.floor, cfg.qom_lo, cfg.qom_hi)
        mid_sh = PT[-1][I_SH].mean(0)
        out.update(activity=activity, qom=qom_s,
                   extent=clip01(np.max(np.linalg.norm(PT[-1][I_WR] - mid_sh, axis=1)) / 2.5))

        m = T >= t - cfg.window_s
        if m.sum() < max(SG_WIN + 3, int(0.6 * cfg.window_s * self.rate)):
            return out
        Pw = PT[m][:, I_WR]                                                      # (n, 2 wrists, 2)
        V = np.stack([sg(Pw[:, w], 1, dt) for w in range(2)], axis=1)
        A = np.stack([sg(Pw[:, w], 2, dt) for w in range(2)], axis=1)
        wspeed = np.linalg.norm(V, axis=2)
        lead = int(np.argmax(wspeed.mean(0)))                                    # the more active wrist
        sp, p, amag = wspeed[:, lead], Pw[:, lead], np.linalg.norm(A[:, lead], axis=1)
        out["speed"] = float(sp[-1])
        out["level"] = clip01(0.5 + (PT[-1][I_WR[lead]][1] - mid_sh[1]) / 4.0)

        # --- layer 3 -------------------------------------------------------------
        a_rms = float(np.sqrt(np.mean(amag ** 2)) + 1e-9)
        lo, hi = np.log10(cfg.acc_lo), np.log10(cfg.acc_hi)
        acc_term = clip01((np.log10(a_rms) - lo) / (hi - lo))

        # Time: how abruptly does speed change?  tangential acceleration / peak speed  [1/s]
        #   sustained ~ <3 1/s   |   sudden ~ >15 1/s   (min-jerk stroke of duration T gives ~3.1/T)
        a_tan = np.einsum("ij,ij->i", V[:, lead], A[:, lead]) / np.maximum(sp, 1e-6)
        moving = sp > 0.15 * sp.max()
        if moving.sum() > 5 and sp.max() > 0.3:
            lam = np.percentile(np.abs(a_tan[moving]), 95) / (np.percentile(sp, 95) + 1e-6)
            impulsivity = clip01((np.log10(max(lam, 1e-3)) - np.log10(cfg.abrupt_lo))
                                 / (np.log10(cfg.abrupt_hi) - np.log10(cfg.abrupt_lo)))
        else:
            impulsivity = 0.0

        sp_now = sp[-1]
        s = self.sparc.push(sp_now)
        if np.isfinite(s):
            self.sparc_value = s
        fluidity = clip01((self.sparc_value - cfg.sparc_bad) / (cfg.sparc_good - cfg.sparc_bad)) \
            if np.isfinite(self.sparc_value) else 0.5

        path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        straight = clip01(np.linalg.norm(p[-1] - p[0]) / path) if path > 0.5 else 0.0

        grounded = None                                   # only when both ankles are in frame
        if vis[[L_AN, R_AN]].min() > 0.5:
            h = PT[-1][I_HIP].mean(0)[1] - PT[-1][I_AN].mean(0)[1]               # ~1.8 torso lengths standing
            crouch = clip01((1.8 - h) / 0.6)
            stance = clip01((np.linalg.norm(PT[-1][I_AN[0]] - PT[-1][I_AN[1]]) - 0.5) / 0.8)
            grounded = 0.6 * crouch + 0.4 * stance
        weight = acc_term if grounded is None else 0.65 * acc_term + 0.35 * grounded

        # repetitivity: autocorrelation of the lead-wrist VELOCITY VECTOR over the long window
        # (speed alone is constant on a circle). Score = height of the first peak after the first
        # trough, so smooth slow signals (which are trivially self-similar at tiny lags) score ~0.
        m4 = T >= t - cfg.long_window_s
        rep = 0.0
        if m4.sum() > 4 * cfg.fps_nominal:
            P4 = PT[m4][:, I_WR[lead]]
            v4 = sg(P4, 1, dt)
            tu = np.arange(T[m4][0], T[m4][-1], 1 / 30.0)
            v4 = np.stack([np.interp(tu, T[m4], v4[:, d]) for d in range(2)], axis=1)
            v4 -= v4.mean(0)
            if np.abs(v4).max() > 0.3:
                ac = sum(np.correlate(v4[:, d], v4[:, d], "full")[len(tu) - 1:] for d in range(2))
                ac = ac / ac[0]
                hi_lag = min(int(3.0 * 30), len(ac) - 2)
                trough = next((i for i in range(2, hi_lag) if ac[i] < ac[i - 1] and ac[i] <= ac[i + 1]), None)
                if trough is not None and trough < hi_lag - 2:
                    rep = clip01(ac[trough:hi_lag].max())

        out.update(weight=weight, time=impulsivity, space=straight, flow=1.0 - fluidity,
                   fluidity=fluidity, impulsivity=impulsivity, repetitivity=rep,
                   sparc=self.sparc_value, grounded=0.0 if grounded is None else grounded)
        return out
