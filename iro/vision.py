"""iro vision: live wrist-camera view with YOLO boxes, in any web browser.

    iro vision            then open http://localhost:8080 on the Jetson,
                          or http://<jetson-ip>:8080 from a laptop on the same network

Shows the stock YOLO classes and every object taught with `iro object add`.
Uses the camera, so stop it (Ctrl+C) before `iro pick` or `iro up camera`.
"""

import http.server
import socket
import socketserver
import sys
import threading
import time

import cv2

from iro.objects import Detector, open_camera

PORT = 8080
PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>IRO wrist camera</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#ddd;font:15px system-ui;text-align:center}
img{max-width:100%;height:auto}p{margin:8px}</style></head>
<body><p>IRO wrist camera + YOLO</p><img src="/stream"></body></html>"""


class Latest:
    """Newest annotated JPEG, shared between the camera loop and the web clients."""

    def __init__(self):
        self.jpg, self.cond = None, threading.Condition()

    def put(self, jpg):
        with self.cond:
            self.jpg = jpg
            self.cond.notify_all()

    def get(self):
        with self.cond:
            self.cond.wait(timeout=2)
            return self.jpg


def camera_loop(latest, stop):
    cap = open_camera()
    det = Detector()
    print(f"Models: {', '.join(det.describe())}")
    n, t0, fps = 0, time.time(), 0.0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        found, annotated = det.detect(frame, draw=True)
        n += 1
        if time.time() - t0 >= 1:
            fps, n, t0 = n / (time.time() - t0), 0, time.time()
        cv2.putText(annotated, f"{fps:.0f} fps", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        latest.put(cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    cap.release()


def main():
    latest, stop = Latest(), threading.Event()
    threading.Thread(target=camera_loop, args=(latest, stop), daemon=True).start()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpg = latest.get()
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif self.path == "/frame.jpg":
                jpg = latest.get()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(jpg or b"")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    ip = socket.gethostbyname(socket.gethostname())
    with Server(("", PORT), Handler) as srv:
        print(f"Live view: http://localhost:{PORT}  (from a laptop: http://{ip}:{PORT})  Ctrl+C to stop")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
