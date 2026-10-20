"""One live camera feed in the browser, shared by every camera program.

`iro vision`, `iro object add` and `iro pick` all call `feed().show(img, text)`;
open http://localhost:8080 once and it shows whatever the robot is doing.
The page reconnects by itself when a program stops and the next one starts.
"""

import http.server
import socket
import socketserver
import threading

import cv2

PORT = 8080
PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>IRO camera</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#ccc;font:15px system-ui;text-align:center}
img{max-width:100%;height:auto;margin-top:6px}p{margin:6px}</style></head>
<body><p>IRO wrist camera &middot; live</p><img id="v" src="/stream">
<p id="s">waiting for a camera program (iro vision, iro object add, iro pick)...</p>
<script>
const v=document.getElementById('v'), s=document.getElementById('s');
v.onload=()=>{s.textContent=''};
v.onerror=()=>{s.textContent='no camera program running - start one (e.g. iro vision); this page reconnects by itself';
  setTimeout(()=>{v.src='/stream?'+Date.now()},2000)};
</script></body></html>"""


class Feed:
    def __init__(self, port=PORT):
        self.jpg, self.cond = None, threading.Condition()
        self.port, self.server = port, None

    def start(self):
        if self.server:
            return self
        feed = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            jpg = feed.wait()
                            if jpg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif self.path.startswith("/frame.jpg"):
                    jpg = feed.jpg or b""
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(jpg)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        try:
            self.server = Server(("", self.port), Handler)
        except OSError:
            print(f"(live feed not started: port {self.port} is busy, another camera program has it)")
            return self
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        try:   # the address other machines reach us on (no packet is actually sent)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
                sk.connect(("10.255.255.255", 1))
                ip = sk.getsockname()[0]
        except OSError:
            ip = "<jetson-ip>"
        print(f"Live feed: http://localhost:{self.port}   (laptop on the same network: http://{ip}:{self.port})")
        return self

    def show(self, img, text=None):
        """Publish a frame (BGR), with an optional status line drawn on it."""
        if text:
            img = img.copy()
            cv2.rectangle(img, (0, img.shape[0] - 28), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
            cv2.putText(img, text[:80], (8, img.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        with self.cond:
            self.jpg = jpg
            self.cond.notify_all()

    def wait(self):
        with self.cond:
            self.cond.wait(timeout=2)
            return self.jpg


_feed = None


def feed() -> Feed:
    """The process-wide feed, started on first use."""
    global _feed
    if _feed is None:
        _feed = Feed().start()
    return _feed
