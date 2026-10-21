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
PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>EFAD live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#ccc;font:15px system-ui;text-align:center}
#w{position:relative;display:inline-block;margin-top:6px;touch-action:none}
img{max-width:100%;height:auto;display:block}
#r{position:absolute;border:3px solid #0f0;display:none;pointer-events:none}p{margin:6px}</style></head>
<body><p>EFAD live &middot; when asked, DRAG a box around the object on the picture</p>
<div id="w"><img id="v" src="/stream" draggable="false"><div id="r"></div></div>
<p id="s">waiting for a camera program (iro efad, iro track, iro object, iro pick)...</p>
<script>
const v=document.getElementById('v'), s=document.getElementById('s'), r=document.getElementById('r');
v.onload=()=>{s.textContent=''};
v.onerror=()=>{s.textContent='no camera program running; this page reconnects by itself';
  setTimeout(()=>{v.src='/stream?'+Date.now()},2000)};
let a=null;
function pos(e){const b=v.getBoundingClientRect(), p=e.touches?e.touches[0]:e;
  return [Math.min(Math.max((p.clientX-b.left)/b.width,0),1), Math.min(Math.max((p.clientY-b.top)/b.height,0),1)]}
function show(p,q){const b=v.getBoundingClientRect();r.style.display='block';
  r.style.left=Math.min(p[0],q[0])*b.width+'px';r.style.top=Math.min(p[1],q[1])*b.height+'px';
  r.style.width=Math.abs(q[0]-p[0])*b.width+'px';r.style.height=Math.abs(q[1]-p[1])*b.height+'px'}
function down(e){e.preventDefault();a=pos(e);show(a,a)}
function move(e){if(a){e.preventDefault();show(a,pos(e))}}
function up(e){if(!a)return;const q=e.changedTouches?(()=>{const b=v.getBoundingClientRect(),t=e.changedTouches[0];
  return [(t.clientX-b.left)/b.width,(t.clientY-b.top)/b.height]})():pos(e);
  const box=[Math.min(a[0],q[0]),Math.min(a[1],q[1]),Math.max(a[0],q[0]),Math.max(a[1],q[1])];a=null;
  if(box[2]-box[0]<0.02||box[3]-box[1]<0.02){r.style.display='none';return}
  fetch('/box',{method:'POST',body:JSON.stringify(box)}).then(()=>{s.textContent='box sent';
  setTimeout(()=>{r.style.display='none'},800)})}
v.addEventListener('mousedown',down);window.addEventListener('mousemove',move);window.addEventListener('mouseup',up);
v.addEventListener('touchstart',down);window.addEventListener('touchmove',move);window.addEventListener('touchend',up);
</script></body></html>"""


class Feed:
    def __init__(self, port=PORT):
        self.jpg, self.cond = None, threading.Condition()
        self.port, self.server = port, None
        self.box, self.box_event = None, threading.Event()    # last box drawn on the page (0-1 coords)

    def start(self):
        if self.server:
            return self
        feed = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                if self.path.startswith("/box"):
                    import json
                    try:
                        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                        feed.box = [float(v) for v in b][:4]
                        feed.box_event.set()
                    except (ValueError, TypeError):
                        pass
                self.send_response(204)
                self.end_headers()

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
        print(f"Live feed: http://localhost:{self.port}  |  laptop on the same Wi-Fi: "
              f"http://{socket.gethostname()}.local:{self.port}  or  http://{ip}:{self.port}")
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

    def ask_box(self, shape, timeout=None):
        """Wait for the person to drag a box on the live page; returns it in pixels (x1,y1,x2,y2)."""
        self.box_event.clear()
        if not self.box_event.wait(timeout):
            return None
        h, w = shape[:2]
        x1, y1, x2, y2 = self.box
        return (x1 * w, y1 * h, x2 * w, y2 * h)

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
