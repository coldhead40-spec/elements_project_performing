"""Find out which camera number is which.   python list_cameras.py

Saves camera_0.jpg, camera_1.jpg ... and, on Windows, prints the NAME of every camera
(install once:  pip install pygrabber)  so you can tell the USB webcam from DroidCam.

BEFORE running:
  1. close TouchDesigner completely (or set its Video Device In to Active = Off)
  2. close Zoom / Teams / Windows Camera / browser tabs that use the camera
  3. DroidCam: start the app on the phone, then in the DroidCam client on the PC press Start
     and check that you SEE the video there.  Leave the client open.
"""
import os
import cv2

try:                                              # camera names, same order as OpenCV's DirectShow indices
    from pygrabber.dshow_graph import FilterGraph
    NAMES = FilterGraph().get_input_devices()
    print("cameras Windows knows about (index = number to use):")
    for n, name in enumerate(NAMES):
        print(f"   {n}: {name}")
    print()
except Exception:
    NAMES = []
    print("(for camera names run:  pip install pygrabber)\n")

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("default", cv2.CAP_ANY)] if os.name == "nt" \
    else [("default", cv2.CAP_ANY)]

for i in range(max(6, len(NAMES))):
    ok, frame, used = False, None, ""
    for label, api in BACKENDS:
        try:
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                for _ in range(20):               # first frames are often empty/dark
                    ok, frame = cap.read()
                cap.release()
                if ok and frame is not None and frame.size:
                    used = label
                    break
                ok = False
        except cv2.error:                         # index does not exist on this PC
            pass
    name = f" ({NAMES[i]})" if i < len(NAMES) else ""
    if ok:
        cv2.imwrite(f"camera_{i}.jpg", frame)
        print(f"camera {i}{name}: OK  {frame.shape[1]}x{frame.shape[0]}  via {used}  -> camera_{i}.jpg")
    else:
        print(f"camera {i}{name}: not available (in use by another program, or not connected)")
