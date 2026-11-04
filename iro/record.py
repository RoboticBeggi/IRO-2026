"""Flight recorder: every program that uses a camera (or other recordable hardware) records
what it saw and what it decided, so any run can be reviewed afterwards.

    iro review [RUN]       contact sheets of every video + summary of logs and events
                           (default: the newest run; RUN = folder name or path)
    iro review list        recent runs

One folder per program run: ~/iro-runs/<time>_<program>/ (~/iro-runs/latest = newest)
    <stream>.mp4      videos: "feed" = everything shown on the live page (recorded
                      automatically by iro.stream), plus any extra stream a program adds
    <table>.csv       numbers per step (joints, scores, decisions...)
    events.txt        timestamped notes: what happened and why
    *.jpg             key pictures
The newest 30 runs are kept (and at most ~4 GB).

In code:  from iro.record import run;  r = run()  (one per process, made on first use)
          r.video("oak", img, ["line 1", "line 2"]);  r.log("poses", pose=3, found=True)
          r.snap("pose03_mask", img);  r.note("gripper not found: too close")
"""

import atexit
import csv
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

RUNS = Path.home() / "iro-runs"
KEEP, MAX_BYTES = 30, 4e9
FPS = 15


def _program():
    a = [x for x in sys.argv[1:] if not x.startswith("-")][:2]
    mod = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] not in ("-c", "-") else "python"
    return "-".join([mod.replace("__main__", "iro")] + a).replace("/", "_")[:60] or "run"


class Run:
    def __init__(self, name=None):
        RUNS.mkdir(exist_ok=True)
        self._prune()
        self.dir = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{name or _program()}"
        self.dir.mkdir(parents=True, exist_ok=True)
        latest = RUNS / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(self.dir.name)
        except OSError:
            pass
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.writers, self.sizes, self.last = {}, {}, {}
        self.tables = {}
        self.note(f"started: {' '.join(sys.argv)}")
        atexit.register(self.close)

    @staticmethod
    def _prune():
        runs = sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink())
        for old in runs[:-(KEEP - 1)] if len(runs) >= KEEP else []:
            shutil.rmtree(old, ignore_errors=True)
        runs = sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink())
        total = sum(f.stat().st_size for p in runs for f in p.iterdir() if f.is_file())
        while runs and total > MAX_BYTES:
            old = runs.pop(0)
            total -= sum(f.stat().st_size for f in old.iterdir() if f.is_file())
            shutil.rmtree(old, ignore_errors=True)

    def t(self):
        return time.time() - self.t0

    def video(self, stream, img, lines=None, fps=FPS):
        """Add a frame to <stream>.mp4 (at most `fps` per second; real time is printed on it)."""
        now = time.time()
        if now - self.last.get(stream, 0) < 1 / fps:
            return
        self.last[stream] = now
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        with self.lock:
            if stream not in self.writers:
                h, w = img.shape[:2]
                self.sizes[stream] = (w, h)
                self.writers[stream] = cv2.VideoWriter(str(self.dir / f"{stream}.mp4"),
                                                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            w, h = self.sizes[stream]
            if img.shape[1] != w or img.shape[0] != h:                  # letterbox to the stream's size
                s = min(w / img.shape[1], h / img.shape[0])
                small = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
                img = np.zeros((h, w, 3), np.uint8)
                img[:small.shape[0], :small.shape[1]] = small
            else:
                img = img.copy()
            for i, txt in enumerate([f"{self.t():7.2f}s"] + [str(x) for x in (lines or []) if x]):
                y = 18 + 20 * i
                cv2.putText(img, txt[:90], (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(img, txt[:90], (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.writers[stream].write(img)

    def log(self, table, **fields):
        """One row in <table>.csv (columns from the first row)."""
        with self.lock:
            if table not in self.tables:
                f = open(self.dir / f"{table}.csv", "w", newline="")
                w = csv.DictWriter(f, fieldnames=["t"] + list(fields), extrasaction="ignore")
                w.writeheader()
                self.tables[table] = (f, w)
            f, w = self.tables[table]
            w.writerow({"t": f"{self.t():.3f}", **{k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in fields.items()}})
            f.flush()

    def snap(self, name, img):
        cv2.imwrite(str(self.dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def note(self, text):
        with open(self.dir / "events.txt", "a") as f:
            f.write(f"{self.t():8.2f}s  {text}\n")

    def close(self):
        with self.lock:
            for w in self.writers.values():
                w.release()
            self.writers.clear()
            for f, _ in self.tables.values():
                f.close()
            self.tables.clear()
        if self.dir.exists() and any(self.dir.iterdir()):
            print(f"Recorded: {self.dir}   (review: iro review)")


_run = None


def run(name=None) -> Run:
    """This process's recording (made on first use)."""
    global _run
    if _run is None:
        _run = Run(name)
    return _run


# ---------------------------------------------------------------- reviewing

def contact_sheets(video, every=0.5, per_sheet=12):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    step = max(1, int(round(fps * every)))
    tiles, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            tiles.append(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))))
        i += 1
    out = []
    for k in range(0, len(tiles), per_sheet):
        group = tiles[k:k + per_sheet]
        group += [np.zeros_like(tiles[0])] * (per_sheet - len(group))
        sheet = np.vstack([np.hstack(group[r * 4:(r + 1) * 4]) for r in range(per_sheet // 4)])
        path = video.with_name(f"{video.stem}_sheet{k // per_sheet:02d}.jpg")
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(path)
    return i, out


def review(which=None):
    if which == "list":
        for p in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink())[-15:]:
            vids = len(list(p.glob("*.mp4")))
            print(f"  {p.name:50s} {vids} video(s)")
        return 0
    d = (RUNS / which if which and (RUNS / which).exists() else Path(which)) if which else (RUNS / "latest").resolve()
    if not d.exists():
        print(f"no run {d}")
        return 1
    print(f"Run {d}")
    for v in sorted(d.glob("*.mp4")):
        n, sheets = contact_sheets(v)
        print(f"  {v.name}: {n} frames -> " + (", ".join(s.name for s in sheets) or "empty"))
    for c in sorted(d.glob("*.csv")):
        rows = list(csv.DictReader(open(c)))
        print(f"  {c.name}: {len(rows)} rows, columns: {', '.join(list(rows[0])[:12]) if rows else '-'}")
    ev = d / "events.txt"
    if ev.exists():
        lines = ev.read_text().splitlines()
        print("  events (last 15):")
        for line in lines[-15:]:
            print("    " + line)
    print(f"  pictures: {', '.join(p.name for p in sorted(d.glob('*.jpg')) if '_sheet' not in p.name)[:400] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(review(sys.argv[1] if len(sys.argv) > 1 else None))
