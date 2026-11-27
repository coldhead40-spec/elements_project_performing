"""python test_poses.py   - sanity tests on synthetic skeletons (proves the plumbing and that the
descriptors separate the four poses; it does NOT prove they match real people - calibrate with main_poses.py)."""
import numpy as np
from pose_match import NAMES, PoseMatcher, synthetic_skeleton

rng = np.random.default_rng(3)
m = PoseMatcher(aspect=16 / 9)
print("pyEyesWeb backend:", m.prim.backend)


def sc(lm, t=1.0):
    f, info = m.analyse(lm, t)
    return m.scores(f), info


def row(name, s):
    print(f"{name:>10}: " + "  ".join(f"{n}={s[n]:.2f}" for n in NAMES))


fails = 0
for jit in (0.0, 0.004):
    for mirror in (False, True):
        print(f"--- jitter {jit}  mirrored {mirror}")
        for pose in NAMES + ["neutral"]:
            s, _ = sc(synthetic_skeleton(pose, mirror=mirror, jitter=jit, rng=rng))
            row(pose, s)
            if pose == "neutral":
                ok = max(s.values()) < 0.4          # below MATCH_ON (0.55) used in TouchDesigner
            else:
                best = max(s, key=s.get)
                ok = best == pose and s[pose] > 0.55 and sorted(s.values())[-2] < 0.35
            fails += not ok
            if not ok:
                print("   ^^^ FAIL")

# legs out of frame: the leg-dependent poses must not match, and full_body must say so
lm = synthetic_skeleton("air")
lm[[25, 26, 27, 28], 3] = 0.1
s, info = sc(lm)
print("legs hidden ->", info["full_body"], f"air={s['air']:.2f}")
fails += not (info["full_body"] == 0.0 and s["air"] < 0.3)

# nobody there
f, info = m.analyse(None, 2.0)
fails += not (f is None and info["present"] == 0.0 and all(v == 0 for v in m.scores(f).values()))

# calibration round-trip: record 'fire' from an arbitrary pose and check it is then recognised
import os, tempfile
p = os.path.join(tempfile.mkdtemp(), "poses.json")
m2 = PoseMatcher(poses_path=p)
feats = [m2.analyse(synthetic_skeleton("air", jitter=0.002, rng=rng), 3 + i / 30)[0] for i in range(60)]
m2.record("fire", feats)   # record fire from the AIR pose
m3 = PoseMatcher(poses_path=p)
f, _ = m3.analyse(synthetic_skeleton("air"), 9.0)
s3 = m3.scores(f)
row("recorded", s3)
fails += not (s3["fire"] > 0.6)

print("ALL OK" if not fails else f"{fails} FAILURES")
raise SystemExit(1 if fails else 0)
