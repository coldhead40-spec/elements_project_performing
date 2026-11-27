"""All tunable numbers live here so you can tune the installation without touching logic."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- I/O ---------------------------------------------------------------
    osc_host: str = "127.0.0.1"
    osc_port: int = 9000
    camera_index: int = 0
    cam_width: int = 1280
    cam_height: int = 720
    mirror: bool = True              # visitors expect a mirror
    model_path: str = "pose_landmarker_lite.task"   # lite = fastest, full = more accurate legs
    prototypes_file: str = "prototypes.json"

    # --- analysis windows ----------------------------------------------------
    window_s: float = 2.0            # effort window (weight, time, space, flow)
    long_window_s: float = 6.0       # repetitivity needs 2+ cycles of a ~3 s rhythm
    fps_nominal: float = 30.0

    # --- landmark filtering (One-Euro) ---------------------------------------
    # Low min_cutoff = smoother but laggy. Higher beta = keeps fast gestures crisp.
    # Too much smoothing destroys "impulsivity", so keep these fairly open.
    oe_min_cutoff: float = 2.0
    oe_beta: float = 0.15

    # --- activity gate (units: torso lengths per second) ----------------------
    qom_noise: float = 0.15          # initial guess of landmark-jitter QoM; adapts while visitor is still
    qom_floor_max: float = 0.35
    floor_rise_s: float = 90.0       # how slowly the floor may creep upward after calibration
    still_calib_s: float = 3.0       # stand still this long at start-up to measure jitter (0 = off)
    qom_lo: float = 0.05             # (QoM - floor) below this = stillness, no element
    qom_hi: float = 0.30             # (QoM - floor) above this = full activity

    # --- feature -> [0,1] mapping ranges (see features.py) -------------------
    acc_lo: float = 1.5              # torso-lengths / s^2  (log scale)
    acc_hi: float = 60.0
    sparc_bad: float = -7.0          # SPARC at/below this -> fluidity 0
    sparc_good: float = -2.0         # SPARC at/above this -> fluidity 1
    abrupt_lo: float = 3.0           # 1/s : speed-change rate that counts as 'sustained' (time=0)
    abrupt_hi: float = 15.0          # 1/s : ... and as 'sudden' (time=1)

    # --- element classifier ---------------------------------------------------
    softmax_temp: float = 0.25
    attack_s: float = 0.30           # how fast an element appears
    release_s: float = 1.80          # how slowly it fades (gives the painting "memory")
    effort_smooth_s: float = 0.40
    out_gain: float = 1.4            # >1 so the mask saturates at strong movement
