"""Camera -> MediaPipe pose -> pyEyesWeb / Laban features -> OSC -> TouchDesigner.

Run:   python main_simple.py                 (camera 0)
       python main_simple.py --camera 1      (another camera; find the number with list_cameras.py)
Stop:  press q in the preview window (or Ctrl+C in the terminal).

Keep this file NEXT TO the 'engine' folder and 'pose_landmarker_lite.task'.
TouchDesigner receives on port 9000 these channels (among others):
    fire, water, air, earth   element weights 0..1
    activity                  0 = standing still, 1 = moving
    dominant                  0 air, 1 earth, 2 fire, 3 water, -1 none
Stand still for the first 3 seconds: that measures the camera jitter.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "2")        # hides MediaPipe's harmless C++ log spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine.config import Config
from engine.elements import ELEMENTS, ElementClassifier
from engine.features import MotionAnalyzer
from engine.osc import OscSender
from engine.pose_source import MediaPipePose

OSC_HOST, OSC_PORT = "127.0.0.1", 9000
KEYS = ["present", "activity", "qom", "speed", "weight", "time", "space", "flow",
        "fluidity", "impulsivity", "repetitivity", "extent", "level",
        "lw_x", "lw_y", "rw_x", "rw_y", "com_x", "com_y"]
COLORS = {"air": (230, 230, 120), "earth": (80, 200, 90), "fire": (60, 90, 255), "water": (255, 160, 60)}


def draw_preview(frame, feat, el):
    h, w = frame.shape[:2]
    frame = cv2.flip(frame, 1)                                   # mirror, like the analysis
    for i, name in enumerate(ELEMENTS):                          # one bar per element
        y = 20 + i * 24
        cv2.rectangle(frame, (10, y), (10 + int(200 * el[name]), y + 16), COLORS[name], -1)
        cv2.putText(frame, name, (220, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if feat.get("present"):
        for k in ("lw", "rw"):                                   # hands
            cv2.circle(frame, (int(feat[k + "_x"] * w), int((1 - feat[k + "_y"]) * h)), 10, (255, 255, 255), 2)
    cv2.putText(frame, f"activity {feat.get('activity', 0):.2f}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="camera index for the pose analysis")
    args = ap.parse_args()
    model = os.path.join(HERE, "pose_landmarker_lite.task")
    if not os.path.exists(model):
        sys.exit(f"Model not found: {model}\nDownload it next to this file (see README, step 1).")

    cfg = Config(osc_host=OSC_HOST, osc_port=OSC_PORT, camera_index=args.camera, model_path=model)
    src = MediaPipePose(cfg)
    analyzer = MotionAnalyzer(cfg, aspect=src.aspect)
    classifier = ElementClassifier(cfg)
    osc = OscSender(cfg.osc_host, cfg.osc_port)
    print(f"smoothness backend: {analyzer.backend}   (pyeyesweb = library in use, numpy-fallback = built-in)")
    print(f"camera {args.camera} | sending OSC to {OSC_HOST}:{OSC_PORT}  -  stand still for 3 s, then move.  q = quit")

    prev_t, last_print, fails = None, 0.0, 0
    try:
        while True:
            r = src.read()
            if r is None:                                        # tolerate a few bad frames at start-up
                fails += 1
                if fails > 60:
                    print(f"No frames from camera {args.camera}. Another program (TouchDesigner's Video Device In?) "
                          "may be using it, or the number is wrong. Run list_cameras.py and use --camera N.")
                    break
                time.sleep(0.03)
                continue
            fails = 0
            t, lm, frame = r
            dt = 1 / 30 if prev_t is None else max(t - prev_t, 1e-3)
            prev_t = t

            feat = analyzer.update(t, lm)
            el = classifier.update(feat, dt)
            msg = {k: feat.get(k, 0.0) for k in KEYS}
            msg.update(el)
            osc.send(msg)                                        # sent every frame, even with nobody in view

            if t - last_print > 1.0:
                last_print = t
                print(f"t={t:6.1f}  person={'yes' if feat.get('present') else 'no '}  "
                      f"activity={feat.get('activity', 0):.2f}  "
                      + "  ".join(f"{e}={el[e]:.2f}" for e in ELEMENTS))

            cv2.imshow("laban - q to quit", draw_preview(frame, feat, el))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
