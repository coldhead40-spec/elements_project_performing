"""Landmark sources: a webcam through MediaPipe Tasks, or a simulated dancer for testing."""
import time
import numpy as np


class MediaPipePose:
    """Webcam -> (t_seconds, landmarks (33,4) | None, bgr_frame).

    Needs the model file once:
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
    """

    def __init__(self, cfg):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.cv2, self.mp = cv2, mp
        self.cap = cv2.VideoCapture(cfg.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cfg.camera_index}")
        opts = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=cfg.model_path),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()
        self.last_ms = -1

    @property
    def aspect(self):
        w = self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH); h = self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)
        return (w / h) if h else 16 / 9

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        t = time.monotonic() - self.t0
        ms = max(int(t * 1000), self.last_ms + 1)           # timestamps must strictly increase
        self.last_ms = ms
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ms)
        lm = None
        if res.pose_landmarks:
            lm = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]])
        return t, lm, frame

    def close(self):
        self.landmarker.close(); self.cap.release()


class SimPose:
    """Synthetic skeleton (virtual clock, no camera). Modes mimic each element's kinematics:
       fire  = abrupt straight strokes     water = big smooth circles
       air   = small drifting wander       earth = slow straight press, low wide stance
       still = nothing; cycle = rotates through the four every `seg` seconds."""
    aspect = 16 / 9
    MODES = ["fire", "water", "air", "earth"]

    def __init__(self, mode="cycle", fps=30.0, seg=12.0, seed=1, jitter=0.0015):
        self.mode, self.fps, self.seg, self.k, self.jitter = mode, fps, seg, 0, jitter
        self.rng = np.random.default_rng(seed)
        self.stroke_dir = np.array([1.0, 0.0]); self.stroke_id = -1

    def current_mode(self, t):
        return self.MODES[int(t // self.seg) % 4] if self.mode == "cycle" else self.mode

    @staticmethod
    def _minjerk(s):
        s = np.clip(s, 0, 1); return 10 * s**3 - 15 * s**4 + 6 * s**5

    def _wrist_offset(self, mode, t):                         # torso lengths, y up, from the shoulder
        if mode == "fire":
            period = 1.0; k = int(t // period); ph = (t % period)
            if k != self.stroke_id:
                self.stroke_id = k; ang = self.rng.uniform(-0.6, 1.2)
                self.stroke_dir = np.array([np.cos(ang), np.sin(ang)])
            out = self._minjerk(ph / 0.15) - self._minjerk((ph - 0.45) / 0.15)
            return self.stroke_dir * 1.3 * out + np.array([0.3, -0.5])
        if mode == "water":
            w = 2 * np.pi / 3.0
            return np.array([0.7 + 0.6 * np.cos(w * t), 0.2 + 0.6 * np.sin(w * t)])
        if mode == "air":
            return np.array([0.5 + 0.22 * np.sin(2 * np.pi * 0.37 * t) + 0.1 * np.sin(2 * np.pi * 0.83 * t + 1),
                             0.3 + 0.22 * np.sin(2 * np.pi * 0.53 * t + 2) + 0.1 * np.sin(2 * np.pi * 0.97 * t)])
        if mode == "earth":
            ph = (t % 5.0) / 5.0
            tri = self._minjerk(ph * 2) if ph < 0.5 else self._minjerk(2 - ph * 2)
            return np.array([0.2, -0.2]) + np.array([1.1, 0.0]) * tri
        return np.array([0.3, -0.9])

    def frame(self, i):
        t = i / self.fps
        mode = self.current_mode(t)
        torso = 0.22
        sh = np.array([[0.40, 0.30], [0.60, 0.30]]); hip_y = 0.30 + torso
        crouch = 0.05 if mode == "earth" else 0.0      # hips drop (image units)
        stance = 0.09 if mode == "earth" else 0.037   # half ankle distance (width units)
        lm = np.zeros((33, 4)); lm[:, 3] = 1.0
        lm[0, :2] = [0.5, 0.2]
        lm[11, :2], lm[12, :2] = sh[0], sh[1]
        lm[23, :2] = [0.44, hip_y + crouch]; lm[24, :2] = [0.56, hip_y + crouch]
        lm[25, :2] = [0.5 - stance * 0.8, hip_y + crouch + 0.19]; lm[26, :2] = [0.5 + stance * 0.8, hip_y + crouch + 0.19]
        lm[27, :2] = [0.5 - stance, 0.92]; lm[28, :2] = [0.5 + stance, 0.92]
        # image-x is mirrored by the analyzer, so build the arm the visitor "raises" on image-right
        off = self._wrist_offset(mode, t) * torso
        wr = sh[1] + np.array([off[0], -off[1]])
        lm[16, :2] = wr; lm[14, :2] = (sh[1] + wr) / 2 + [0.0, 0.03]
        lm[15, :2] = sh[0] + [-0.03, 0.18]; lm[13, :2] = sh[0] + [-0.02, 0.09]
        lm[:, :2] += self.rng.normal(0, self.jitter, (33, 2))      # webcam-ish jitter
        return t, lm, mode
