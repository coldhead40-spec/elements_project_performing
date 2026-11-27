# Four elements — pose version (2 cameras · MediaPipe · pyEyesWeb · TouchDesigner)

```
camera 1 ─► Video Device In (TD) ─► flip ─► threshold/blur ─► level_shadow ─┐  your shadow
camera 2 ─► main_poses.py ─► MediaPipe ─► pose_match.py ─► OSC :9000 ─► oscin1 ─► pose_logic
                                                                              │ (state machine)
 avatar videos ─► switch1 ─► level_shadow1 ──────────────────────────────────┤  the pose to copy
 mono1 + 4 element videos ─► reveal (GLSL) ◄── progress per element ──────────┤  zones awaken
 pose_text, snd_* loops ◄─────────────────────────────────────────────────────┘
                              reveal + shadow + avatar + text ─► pose_comp ─► pose_out ─► out1
```

## The four poses (read off your avatar videos)
| Element | What the avatar does | What the engine measures |
|---|---|---|
| Water | arms stretched out to the sides, loose, flowing; legs together | arms ~100° from the torso, wide hand gap, hands at shoulder height |
| Earth | mountain pose: standing tall, feet together, arms hanging | arms ~172° (down), straight legs, **and steady** |
| Air | tree pose, hands on hips, one foot against the standing leg | elbows ~95°, one ankle lifted ~0.95 torso lengths, bent knee |
| Fire | both arms overhead, elbows out, hands joined above the head | arms ~55° from the torso, hands together and above the shoulders |

Earth is basically *standing still*, so a relaxed visitor would match it by accident. `pose_logic` therefore only
counts earth while `steady` > 0.4 (`NEED_STEADY`). Expect it to be the easiest pose; if that bothers you, raise the value.

## Run
1. Put `main_poses.py`, `pose_match.py`, `test_poses.py` next to `main_simple.py` (the `engine` folder and
   `pose_landmarker_lite.task` are reused). `python list_cameras.py` to find the camera numbers.
2. `python test_poses.py` → must print `ALL OK`.
3. Without a camera first: `python main_poses.py --sim cycle` (a fake visitor does the four poses in turn).
   With the camera: `python main_poses.py --camera 1 --backend msmf` (use the camera that is NOT in TouchDesigner's Video Device In;
   `--backend msmf` is needed for DroidCam, a USB webcam works with `--backend dshow` or without the option).
   The window prints `pyEyesWeb backend: pyeyesweb` or `numpy-fallback`.
4. TouchDesigner: paste `experience_poses_setup.py` into a Text DAT → Run Script once. Read the console for `!!` lines
   (parameter names that differ in your build). Put `water.wav earth.wav air.wav fire.wav` in a `sounds/` folder next to the .toe.
5. Test the show without anybody: `op('pose_logic').module.complete('water')`, `.reset()`.

## Learn the poses from the videos (optional, do this first)
Put the four `*_avatar.mp4` next to the scripts and run `python poses_from_videos.py` (needs `pose_landmarker_lite.task`).
It writes `poses.json` and prints a 4x4 table: each video against each template. The diagonal must be > 0.55, the rest low.
MediaPipe was trained on photos, not flat silhouettes, so check the table; if a pose looks wrong, delete `poses.json`
(back to the built-in defaults) or use the live calibration below.

## Calibrate (the important step)
Copy each avatar yourself in front of camera 2, and press `1` water, `2` earth, `3` air, `4` fire; hold ~2 s each.
This writes `poses.json` (delete it to return to the defaults). Then watch the four bars: the right one should be > 0.55
while you hold the pose and the others < 0.25. `--tol 1.3` makes everything more forgiving.
Tune in `pose_logic`: `MATCH_ON/OFF`, `HOLD` (seconds to fill a zone), `POSE_TIMEOUT`. Tune the zones in `reveal_glsl` (`Z_*`).

## Not tested
Tested here: the descriptors/scoring/calibration on synthetic skeletons (`test_poses.py`; the skeletons are drawn by me to look like the avatars) and the state machine with a mocked
TouchDesigner. NOT tested: live MediaPipe, the real pyEyesWeb package (the adapter falls back silently — check the printed backend),
`poses_from_videos.py` on the real mp4s (no pose model file in my environment), and every TouchDesigner node/parameter name and the GLSL.
