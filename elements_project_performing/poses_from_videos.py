"""Learn the four pose templates from the avatar videos themselves.

    python poses_from_videos.py                     # looks for water_avatar.mp4 ... next to this file
    python poses_from_videos.py --videos C:\\path\\to\\avatars

Runs MediaPipe on the hold part of each video, measures the same descriptors the live engine uses
(pose_match.py) and writes poses.json.  Then prints, for every video, how well it matches each of the
four templates: the diagonal (water-water, earth-earth ...) should be > 0.55 and the rest low.

CAUTION: MediaPipe was trained on photos of people, not flat silhouettes, so an arm may be placed wrongly
(typical: the fire "diamond" arms, or the hands-on-hips of air).  If the printed matrix is bad, either
edit poses.json by hand, or do the reliable thing: stand in each pose yourself and press 1-4 in
main_poses.py.  Delete poses.json to return to the built-in defaults.
Keep this file next to pose_match.py and pose_landmarker_lite.task.
"""
import argparse
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pose_match import NAMES, PoseMatcher

# fraction of each video where the pose is being held (read off the frames: arms/legs are in place)
WINDOW = {"water": (0.25, 0.85), "earth": (0.10, 0.90), "air": (0.25, 0.72), "fire": (0.30, 0.72)}


def landmarks_per_frame(path, model, lo, hi):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    lm_task = vision.PoseLandmarker.create_from_options(opts)
    out, total = [], 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if not (lo * n <= i <= hi * n):
            continue
        total += 1
        res = lm_task.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            int(i * 1000 / fps) + 1)
        if res.pose_landmarks:
            out.append((i / fps, np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]])))
    lm_task.close()
    cap.release()
    return out, total, w / h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=HERE, help="folder with water_avatar.mp4, earth_avatar.mp4, ...")
    ap.add_argument("--model", default=os.path.join(HERE, "pose_landmarker_lite.task"))
    args = ap.parse_args()
    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}\nDownload it next to this file (see README, step 1).")

    out_path = os.path.join(HERE, "poses.json")
    if os.path.exists(out_path):
        os.remove(out_path)                                    # start from scratch
    frames = {}
    for name in NAMES:
        path = os.path.join(args.videos, f"{name}_avatar.mp4")
        lo, hi = WINDOW[name]
        seq, total, aspect = landmarks_per_frame(path, args.model, lo, hi)
        print(f"{name:6}: MediaPipe found a person in {len(seq)}/{total} frames of the hold phase")
        frames[name] = (seq, aspect)
        if len(seq) < 10:
            print(f"   !! too few detections for {name} - MediaPipe does not like this silhouette. "
                  "Use the live calibration (keys 1-4 in main_poses.py) for this pose.")

    matcher = PoseMatcher(aspect=frames["water"][1], poses_path=out_path)
    feats = {}
    for name in NAMES:
        seq, aspect = frames[name]
        matcher.aspect = aspect
        matcher._prev = None
        feats[name] = [f for f in (matcher.analyse(lm, t)[0] for t, lm in seq) if f is not None]
        if len(feats[name]) >= 10:
            tpl = matcher.record(name, feats[name])
            print(f"{name:6}: " + "  ".join(f"{k}={v[0]:.2f}" for k, v in tpl.items()))

    print("\nhow each video scores against each template (rows = video, columns = template):")
    print("          " + "".join(f"{n:>8}" for n in NAMES))
    for name in NAMES:
        if not feats[name]:
            continue
        mean = {n: np.mean([matcher.scores(f)[n] for f in feats[name]]) for n in NAMES}
        print(f"{name:>8}  " + "".join(f"{mean[n]:8.2f}" for n in NAMES))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
