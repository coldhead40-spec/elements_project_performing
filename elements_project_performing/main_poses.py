"""Camera 2 -> MediaPipe pose -> pose matching (pyEyesWeb primitives) -> OSC -> TouchDesigner.

Run:   python main_poses.py --camera 1 --backend msmf   (the camera that WATCHES the visitor's pose;
                                                          DroidCam needs --backend msmf, a USB webcam works with dshow/auto)
       python main_poses.py --sim cycle           (no camera: a fake visitor cycles through the 4 poses)
Keys in the preview window:
       q  quit
       1 2 3 4   record the pose you are standing in as  water / earth / air / fire  (hold it ~2 s)
                 -> writes poses.json next to this file; delete the file to go back to the defaults

Keep this file and pose_match.py NEXT TO the 'engine' folder and 'pose_landmarker_lite.task'
(same place as main_simple.py).  Camera 1 (the shadow) is NOT opened here - TouchDesigner's
Video Device In uses it.  Use list_cameras.py to find the numbers.

OSC channels sent to TouchDesigner (port 9000), every frame:
    present, full_body, steady, balance        0/1 flags and 0..1 values
    match_water, match_earth, match_air, match_fire     0..1 how well the visitor matches each pose
    best                                       index in water,earth,air,fire order, -1 = nothing above 0.3
"""
import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # makes the MSMF backend (DroidCam) start fast

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine.osc import OscSender                       # same sender main_simple.py uses
from pose_match import NAMES, PoseMatcher, synthetic_skeleton

OSC_HOST, OSC_PORT = "127.0.0.1", 9000
SMOOTH_TAU = 0.25                                      # s, smoothing of the match scores
RECORD_SECONDS = 2.0
COLORS = {"water": (255, 160, 60), "earth": (60, 120, 180), "air": (230, 230, 230), "fire": (60, 90, 255)}
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28)]


def force_backend(name):
    """Make every cv2.VideoCapture(index, ...) used by engine/pose_source.py open with the chosen backend.
    (OpenCV can open DroidCam only through MSMF; the default DSHOW attempt fails on it.)"""
    api = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}[name]
    real = cv2.VideoCapture

    def capture(*a, **k):
        try:
            return real(a[0], api)
        except cv2.error:
            return real()                                  # an unopened capture -> pose_source reports it
    cv2.VideoCapture = capture


def draw_preview(frame, lm, scores, info, status):
    h, w = frame.shape[:2]
    frame = cv2.flip(frame, 1)                                       # mirror, like the visitor sees it
    if lm is not None:
        pts = [(int((1 - x) * w), int(y * h)) for x, y, _, _ in lm]
        for a, b in BONES:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for i, n in enumerate(NAMES):
        y = 20 + i * 24
        cv2.rectangle(frame, (10, y), (10 + int(200 * scores[n]), y + 16), COLORS[n], -1)
        cv2.rectangle(frame, (10, y), (210, y + 16), (255, 255, 255), 1)
        cv2.putText(frame, f"{n} {scores[n]:.2f}", (220, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    note = status or ("step back: legs not visible" if info["present"] and not info["full_body"] else "")
    cv2.putText(frame, note or f"steady {info['steady']:.2f}  balance {info['balance']:.2f}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if note else (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="camera index for the pose analysis (camera 2)")
    ap.add_argument("--backend", choices=["auto", "msmf", "dshow"], default="auto",
                    help="how OpenCV opens the camera: msmf for DroidCam, dshow for most USB webcams")
    ap.add_argument("--sim", choices=["cycle", "neutral"] + NAMES, help="no camera: simulated visitor")
    ap.add_argument("--seg", type=float, default=9.0, help="--sim cycle: seconds per pose")
    ap.add_argument("--tol", type=float, default=1.4, help="tolerance multiplier (>1 = more forgiving, 1.0 = strict)")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    src = None
    if args.backend != "auto" and os.name == "nt":
        force_backend(args.backend)
    if args.sim:
        aspect = 16 / 9
    else:
        model = os.path.join(HERE, "pose_landmarker_lite.task")
        if not os.path.exists(model):
            sys.exit(f"Model not found: {model}\nDownload it next to this file (see README, step 1).")
        from engine.pose_source import MediaPipePose
        src = MediaPipePose(SimpleNamespace(camera_index=args.camera, cam_width=640, cam_height=480,
                                            model_path=model))
        aspect = src.aspect

    matcher = PoseMatcher(aspect=aspect, poses_path=os.path.join(HERE, "poses.json"), tol_scale=args.tol)
    osc = OscSender(OSC_HOST, OSC_PORT)
    print(f"pyEyesWeb backend: {matcher.prim.backend}   (pyeyesweb = library in use, numpy-fallback = built-in)")
    print("poses from poses.json:", matcher.custom or "none (defaults)")
    print(f"{'sim ' + args.sim if args.sim else 'camera ' + str(args.camera)} | OSC -> {OSC_HOST}:{OSC_PORT} | "
          "keys: 1-4 record water/earth/air/fire, q quit")

    smooth = {n: 0.0 for n in NAMES}
    rec = None                                  # (name, t_end, [features])
    status, status_until = "", 0.0
    t0 = time.monotonic()
    prev_t, last_print, fails, i = None, 0.0, 0, 0
    rng = np.random.default_rng(1)
    try:
        while True:
            if args.sim:
                t = time.monotonic() - t0
                pose = NAMES[int(t // args.seg) % 4] if args.sim == "cycle" else args.sim
                if args.sim == "cycle" and (t % args.seg) < 1.0:
                    pose = "neutral"                                # a second of standing between poses
                lm, frame = synthetic_skeleton(pose, aspect, jitter=0.002, rng=rng), np.zeros((360, 640, 3), np.uint8)
                time.sleep(1 / 30)
            else:
                r = src.read()
                if r is None:
                    fails += 1
                    if fails > 60:
                        print(f"No frames from camera {args.camera}. Is TouchDesigner's Video Device In using it? "
                              "Run list_cameras.py and pass --camera N.")
                        break
                    time.sleep(0.03)
                    continue
                fails = 0
                t, lm, frame = r
            dt = 1 / 30 if prev_t is None else max(t - prev_t, 1e-3)
            prev_t = t

            feat, info = matcher.analyse(lm, t)
            raw = matcher.scores(feat)
            k = 1 - math.exp(-dt / SMOOTH_TAU)
            for n in NAMES:
                smooth[n] += (raw[n] - smooth[n]) * k

            if rec is not None and feat is not None:
                rec[2].append(feat)
                status = f"recording {rec[0]} ... {max(0.0, rec[1] - time.monotonic()):.1f}s"
                if time.monotonic() >= rec[1]:
                    tpl = matcher.record(rec[0], rec[2])
                    status, status_until = f"saved {rec[0]} ({len(tpl)} descriptors) -> poses.json", time.monotonic() + 3
                    print(status)
                    rec = None
            elif rec is not None:
                rec = (rec[0], time.monotonic() + RECORD_SECONDS, [])    # waits until a person is seen
                status = f"recording {rec[0]}: stand in view"
            elif time.monotonic() > status_until:
                status = ""

            best = max(NAMES, key=smooth.get)
            msg = dict(info)
            msg.update({"match_" + n: smooth[n] for n in NAMES})
            msg["best"] = float(NAMES.index(best)) if smooth[best] > 0.3 else -1.0
            osc.send(msg)

            if t - last_print > 1.0:
                last_print = t
                print(f"t={t:6.1f} person={'yes' if info['present'] else 'no '} legs={'yes' if info['full_body'] else 'no '} "
                      + "  ".join(f"{n}={smooth[n]:.2f}" for n in NAMES))

            if not args.no_preview:
                cv2.imshow("poses - q quit, 1-4 record", draw_preview(frame, lm, smooth, info, status))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("1"), ord("2"), ord("3"), ord("4")) and not args.sim:
                    rec = (NAMES[key - ord("1")], time.monotonic() + RECORD_SECONDS, [])
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        if src is not None:
            src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
