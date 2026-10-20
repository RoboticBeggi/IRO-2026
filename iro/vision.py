"""iro vision: live wrist-camera view with YOLO boxes, in any web browser.

    iro vision            then open http://localhost:8080 on the Jetson,
                          or http://<jetson-ip>:8080 from a laptop on the same network

Green: stock YOLO classes. Orange: trained objects (`iro object train`). Pink: remembered
objects (`iro object show`, instant).
`iro object add` and `iro pick` show their own view on the same page while they run.
"""

import sys
import time

from iro.objects import Detector, open_camera


def main():
    det = Detector()
    cam = open_camera()
    print(f"Models: {', '.join(det.describe())}.  Ctrl+C to stop.")
    stats = {"n": 0, "t": time.time(), "fps": 0.0}

    def overlay(frame):
        found, img = det.detect(frame, draw=True)
        stats["n"] += 1
        if time.time() - stats["t"] >= 1:
            stats["fps"], stats["n"], stats["t"] = stats["n"] / (time.time() - stats["t"]), 0, time.time()
        names = ", ".join(f"{n} {p:.2f}" for n, p, *_ in found[:4]) or "nothing recognised"
        return img, f"{stats['fps']:.0f} fps | {names}"

    cam.overlay = overlay
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
