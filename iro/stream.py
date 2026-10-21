"""One live camera page in the browser that is ALWAYS there, for every camera program.

    iro feed            status (and start it if it is not running)
    iro feed stop | restart
    iro feed install    also start it at every boot (user crontab @reboot)

The page is served by a small helper that keeps running on its own (it is started by the first
camera program that needs it, and at boot by cron @reboot). Programs (`iro track`, `iro efad`,
`iro object`, `iro pick`, ...) send their pictures to it with `feed().show(img, text)`. When no
program runs, the page shows a standby picture instead of disappearing; it checks in every
second and reconnects by itself, so it never needs a refresh.

    http://localhost:8080   |   laptop: http://ubuntu.local:8080 (needs: sudo ufw allow 8080/tcp)
"""

import http.client
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

PORT = 8080
LOG = os.path.expanduser("~/.iro-feed.log")
IDLE_AFTER = 2.0          # s without a picture from a program = standby
PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>EFAD live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#ccc;font:15px system-ui;text-align:center}
#w{position:relative;display:inline-block;margin-top:6px;touch-action:none}
img{max-width:100%;height:auto;display:block}
#r{position:absolute;border:3px solid #0f0;display:none;pointer-events:none}p{margin:6px}
#ask{display:none;background:#0a0;color:#fff;padding:6px;font-weight:bold}
#dot{display:inline-block;width:10px;height:10px;border-radius:5px;background:#555;margin-right:6px}</style></head>
<body><p><span id="dot"></span><span id="s">connecting...</span></p>
<div id="ask">DRAG a box around the object on the picture</div>
<div id="w"><img id="v" src="/stream" draggable="false"><div id="r"></div></div>
<script>
const v=document.getElementById('v'), s=document.getElementById('s'), r=document.getElementById('r'),
      dot=document.getElementById('dot'), ask=document.getElementById('ask');
let boot=null, down_=false;
function reconnect(){v.src='/stream?'+Date.now()}
v.onerror=()=>{setTimeout(reconnect,1000)};
async function beat(){            // every second: is the helper there, did it restart, what runs?
  try{const st=await (await fetch('/status',{cache:'no-store'})).json();
    if(down_||(boot&&st.boot!==boot)){reconnect()}
    boot=st.boot; down_=false;
    dot.style.background=st.live?'#0c0':'#c90';
    s.textContent=st.live?(st.text||'live'):'standby: no camera program running (start one, e.g. iro track Tape)';
    ask.style.display=st.asking?'block':'none';
  }catch(e){down_=true;dot.style.background='#c00';s.textContent='robot not reachable, retrying...'}
}
setInterval(beat,1000);beat();
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


def links():
    try:   # the address other machines reach us on (no packet is actually sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            ip = sk.getsockname()[0]
    except OSError:
        ip = "<jetson-ip>"
    return (f"Live feed: http://localhost:{PORT}  |  laptop on the same Wi-Fi: "
            f"http://{socket.gethostname()}.local:{PORT}  or  http://{ip}:{PORT}")


# ---------------------------------------------------------------- the always-on helper (hub)

class Hub:
    def __init__(self):
        self.jpg, self.text, self.t_push = None, "", 0.0
        self.cond = threading.Condition()
        self.box, self.box_event, self.asking = None, threading.Event(), False
        self.boot = str(time.time())

    def standby(self):
        """Last picture, dimmed, with a note: shown while no program sends pictures."""
        img = None
        if self.jpg is not None:
            img = cv2.imdecode(np.frombuffer(self.jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((480, 640, 3), np.uint8)
        img = (img * 0.35).astype(np.uint8)
        for i, t in enumerate(("STANDBY - no camera program running",
                               "start one, e.g.  iro track Tape", time.strftime("%H:%M:%S"))):
            cv2.putText(img, t, (20, 200 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def live(self):
        return time.time() - self.t_push < IDLE_AFTER

    def next_jpg(self, last_sent):
        """Block until a new picture (or 0.5 s in standby)."""
        with self.cond:
            self.cond.wait(timeout=0.5 if not self.live() else 2)
            if self.live() and self.jpg is not None:
                return self.jpg if self.jpg is not last_sent else None
            return self.standby()

    def serve(self):
        hub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def body(self):
                return self.rfile.read(int(self.headers.get("Content-Length", 0)))

            def reply(self, code, data=b"", ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path.startswith("/push"):            # a program sends a picture
                    jpg = self.body()
                    with hub.cond:
                        hub.jpg, hub.t_push = jpg, time.time()
                        hub.text = self.headers.get("X-Text", "")
                        hub.cond.notify_all()
                    self.reply(204)
                elif self.path.startswith("/box"):           # the person drew a box
                    try:
                        hub.box = [float(v) for v in json.loads(self.body())][:4]
                        hub.box_event.set()
                    except (ValueError, TypeError):
                        pass
                    self.reply(204)
                else:
                    self.reply(404)

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            jpg = hub.next_jpg(last)
                            if jpg:
                                last = jpg
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif self.path.startswith("/status"):
                    self.reply(200, json.dumps({"boot": hub.boot, "live": hub.live(), "text": hub.text,
                                                "asking": hub.asking}).encode())
                elif self.path.startswith("/askbox"):        # a program waits for a drawn box
                    timeout = float(self.path.split("timeout=")[1]) if "timeout=" in self.path else 120
                    hub.box_event.clear()
                    hub.asking = True
                    got = hub.box_event.wait(timeout)
                    hub.asking = False
                    self.reply(200, json.dumps(hub.box if got else None).encode())
                elif self.path.startswith("/frame.jpg"):
                    self.reply(200, hub.jpg if hub.live() and hub.jpg else hub.standby(), "image/jpeg")
                else:
                    self.reply(200, PAGE, "text/html")

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        Server(("", PORT), Handler).serve_forever()


def hub_alive(timeout=0.3):
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        c.request("GET", "/status")
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except OSError:
        return False


def start_hub(wait=8.0):
    """Start the helper in the background (it outlives the program that started it)."""
    if hub_alive():
        return True
    with open(LOG, "a") as log:
        subprocess.Popen([sys.executable, "-m", "iro.stream", "serve"], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    t0 = time.time()
    while time.time() - t0 < wait:
        if hub_alive():
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------- what programs use

class Feed:
    """Sends pictures to the helper (only the newest one: a slow network never slows the robot)."""

    def __init__(self):
        self.pending, self.cond, self.ok = None, threading.Condition(), False

    def start(self):
        self.ok = start_hub()
        print(links() if self.ok else f"(live page could not start, see {LOG})")
        threading.Thread(target=self._sender, daemon=True).start()
        return self

    def _sender(self):
        conn = None
        while True:
            with self.cond:
                while self.pending is None:
                    self.cond.wait()
                jpg, text = self.pending
                self.pending = None
            for _ in range(2):
                try:
                    if conn is None:
                        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
                    conn.request("POST", "/push", body=jpg, headers={"X-Text": text.encode("ascii", "replace").decode()})
                    conn.getresponse().read()
                    break
                except (OSError, http.client.HTTPException):
                    conn = None
                    if not hub_alive():
                        start_hub(wait=3)                   # helper died: bring it back

    def show(self, img, text=None):
        """Publish a frame (BGR), with an optional status line drawn on it."""
        if text:
            img = img.copy()
            cv2.rectangle(img, (0, img.shape[0] - 28), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
            cv2.putText(img, text[:80], (8, img.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        with self.cond:
            self.pending = (jpg, (text or "")[:200])
            self.cond.notify()

    def ask_box(self, shape, timeout=None):
        """Wait for the person to drag a box on the live page; returns it in pixels (x1,y1,x2,y2)."""
        timeout = 3600 if timeout is None else timeout
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout + 5)
            c.request("GET", f"/askbox?timeout={timeout}")
            box = json.loads(c.getresponse().read() or b"null")
            c.close()
        except (OSError, ValueError, http.client.HTTPException):
            return None
        if not box:
            return None
        h, w = shape[:2]
        x1, y1, x2, y2 = box
        return (x1 * w, y1 * h, x2 * w, y2 * h)


_feed = None


def feed() -> Feed:
    """The process-wide feed, started on first use."""
    global _feed
    if _feed is None:
        _feed = Feed().start()
    return _feed


def hub_pids():
    out = subprocess.run(["pgrep", "-f", "iro.stream serve"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "serve":
        try:
            Hub().serve()
        except OSError as e:
            print(f"port {PORT} busy ({e}): another helper is already serving")
        return 0
    if cmd == "install":
        iro = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "iro")
        line = f"@reboot sleep 20 && {iro} feed >> {LOG} 2>&1"
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
        if line not in cur:
            keep = [x for x in cur.splitlines() if " feed >> " not in x]
            subprocess.run(["crontab", "-"], input="\n".join(keep + [line]) + "\n", text=True, check=True)
        print("live page helper starts at every boot (crontab @reboot). Remove: crontab -e")
    if cmd in ("stop", "restart"):
        import signal
        for pid in hub_pids():
            os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        if cmd == "stop":
            print("live page helper stopped")
            return 0
    ok = start_hub()
    print(("live page helper running (pid " + ", ".join(map(str, hub_pids())) + ")") if ok
          else f"could not start the live page helper, see {LOG}")
    print(links())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
