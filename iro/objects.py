"""Teach the robot your own objects: the arm films them, labels them, trains YOLO.

    iro object show wayang     INSTANT: show it to the camera ~15 s, usable at once (no training)
    iro object add wayang      record a new object called "wayang" (arm does the angles)
    iro object list            taught objects and how many pictures each has
    iro object train           train YOLO on all taught objects (~10-20 min on the Jetson)
    iro object relabel wayang  re-do the boxes of existing pictures with SAM
    iro object remove wayang

How `add` works (no drawing boxes by hand):
  1. clear the work area: the arm visits a set of camera poses and photographs the empty table
  2. put the object down: the arm visits the SAME poses again, over several rounds;
     between rounds you turn the object a quarter turn (or lay it on another side)
  3. each photo minus the empty-table photo from the same pose = where the object is,
     so the training boxes draw themselves. A contact sheet shows them for checking.
Pictures live in objects/<name>/ (backed up with the repo); the trained model is
objects/model/objects.pt and is used by `iro vision` and `iro pick` automatically.
"""

import json
import shutil
import sys
import time

import cv2
import numpy as np

from iro import ROOT, load_config

OBJ_DIR = ROOT / "objects"
OBJ_MODEL = OBJ_DIR / "model" / "objects.pt"
STOCK_MODEL = ROOT / "models" / "yolo11n.pt"
SHEET = OBJ_DIR / "{name}" / "check.jpg"


class Camera:
    """Wrist camera that keeps the browser feed (iro.stream) live in the background, even while
    the program waits for Enter or the arm moves. `overlay(frame) -> (frame, text)` decorates it."""

    def __init__(self):
        import threading

        from iro.stream import feed
        dev = load_config()["devices"]["arm_camera"]["device"]
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open the wrist camera {dev}: another program has it "
                               "(iro vision / iro pick / iro object / iro up camera). Stop that one first.")
        self.lock = threading.Lock()
        self.feed = feed()
        self.overlay = None
        self.status = ""
        self.hold_until = 0.0
        self.running = True
        threading.Thread(target=self._preview, daemon=True).start()

    def _preview(self):
        while self.running:
            with self.lock:
                ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            text = self.status
            if self.overlay:
                try:
                    f, extra = self.overlay(f)
                    text = " | ".join(t for t in (self.status, extra) if t)
                except Exception:  # noqa: BLE001 - a drawing problem must not stop the feed
                    pass
            if time.time() >= self.hold_until:
                self.feed.show(f, text)
            if not self.overlay:
                time.sleep(0.03)   # plain preview: ~30 fps is plenty; with YOLO the detector sets the pace

    def flash(self, img, text, seconds=0.6):
        """Show a still (e.g. a just-labelled picture) for a moment before the live view resumes."""
        self.feed.show(img, text)
        self.hold_until = time.time() + seconds

    def lock_exposure(self, target=120):
        """Fix the exposure: auto-exposure re-adjusts when someone moves nearby, which changes the
        whole picture and spoils the empty-table comparison. First steer it so the picture
        averages `target` (mid-grey): a white object on a washed-out table has no contrast."""
        with self.lock:
            exp = max(self.cap.get(cv2.CAP_PROP_EXPOSURE), 1.0)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # V4L2: 1 = manual, 3 = auto
            self.exposure_locked = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1
        if not self.exposure_locked:
            return False
        for _ in range(5):
            with self.lock:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            time.sleep(0.3)
            mean = float(grab(self).mean())
            if abs(mean - target) < 12:
                break
            exp = float(np.clip(exp * target / max(mean, 1.0), 1, 5000))
        self.exposure_value = exp
        return True

    def read(self):
        with self.lock:
            return self.cap.read()

    def release(self):
        self.running = False
        time.sleep(0.1)
        with self.lock:
            if getattr(self, "exposure_locked", False):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # back to auto for everyone else
            self.cap.release()


def open_camera():
    return Camera()


def grab(cam, n=1):
    """Fresh frame(s): drop the ones buffered while the arm moved; median of n reduces noise."""
    with cam.lock:
        for _ in range(4):
            cam.cap.grab()
        frames = [f for ok, f in (cam.cap.read() for _ in range(n)) if ok]
    if not frames:
        raise RuntimeError("wrist camera gives no image")
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if n > 1 else frames[0]


def draw_box(img, box, color=(0, 200, 255), label=None):
    img = img.copy()
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        if label:
            cv2.putText(img, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


class Detector:
    """Stock YOLO (80 everyday classes) + the model trained on your own objects, together."""

    def __init__(self, conf=0.4):
        from ultralytics import YOLO
        self.conf = conf
        self.models = [("stock", YOLO(str(STOCK_MODEL)))]
        if OBJ_MODEL.exists():
            self.models.append(("taught", YOLO(str(OBJ_MODEL))))
        # Instant recognition (iro object show): no training, works as soon as something is remembered
        from iro.instant import MEM, Recognizer
        self.instant = Recognizer() if any(p for p in MEM.glob("*.npy") if not p.stem.startswith("_")) else None

    def describe(self):
        out = ["stock YOLO (80 classes)"]
        if len(self.models) > 1:
            out.append("trained: " + ", ".join(self.models[1][1].names.values()))
        if self.instant:
            out.append("remembered: " + ", ".join(self.instant.memory.objects))
        return out

    @property
    def taught_names(self):
        names = list(self.models[1][1].names.values()) if len(self.models) > 1 else []
        return names + (list(self.instant.memory.objects) if self.instant else [])

    def detect(self, frame, conf=None, draw=False):
        """[(name, conf, (u, v) centre, box xyxy, source)], best first; plus an annotated copy if draw."""
        found = []
        for source, m in self.models:
            r = m.predict(frame, device=0, verbose=False, conf=conf or self.conf)[0]
            for b, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls, r.boxes.conf):
                found.append((r.names[int(c)], float(p), np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), b, source))
        if self.instant:
            for name, margin, c, b in self.instant.recognize(frame):
                found.append((name, min(1.0, 0.5 + margin), c, b, "instant"))   # margin > -0.1 => score > 0.4
        found.sort(key=lambda d: -d[1])
        if not draw:
            return found, None
        img = frame.copy()
        for name, p, _, b, source in found:
            color = {"taught": (0, 200, 255), "instant": (255, 0, 255)}.get(source, (0, 255, 0))
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} {p:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return found, img


# --- capture -----------------------------------------------------------------

def capture_poses(arm, cfg):
    """Camera poses around the look point: [(xyz, joint angles)], solved ONCE from fixed starting
    points and replayed exactly (re-solving from wherever the arm happens to be lands a few pixels
    differently, which breaks the empty-table comparison).

    Variety for training, within what a 5-joint arm can do: it cannot swing round to look at
    the object from its side (the gripper always points within the vertical plane through the
    base), so the SIDES come from you turning the object between rounds, and the arm supplies
    DISTANCE (zoom: 10-24 cm above the table), left/right shifts (+-8 cm) and lean."""
    from iro.motion import SEEDS
    lx, ly = cfg["look_xy"]
    roll = arm.level_roll()
    poses = []
    for k, height in enumerate((0.10, 0.14, 0.19, 0.24)):
        for j, dy in enumerate((-0.08, -0.04, 0.0, 0.04, 0.08)):
            dx = (-0.03, 0.03)[(j + k) % 2] if dy else 0.0     # alternate near/far for more variety
            xyz = [lx + dx, ly + dy, cfg["table_z"] + height]
            for lean in (0, -10, -20, -30, -40):
                tilt = np.radians(cfg["look_tilt_deg"] + lean)
                q, err = min((arm.kin.ik_down(xyz, np.r_[np.asarray(s)[:4], roll], tilt=tilt) for s in SEEDS),
                             key=lambda r: r[1])
                if err < 0.004:
                    poses.append((xyz, q))
                    break
    return poses


class Refiner:
    """Tighten a rough box with Segment Anything (MobileSAM, box prompt). The empty-table
    difference misses parts of an object that look like the table; SAM recovers the whole
    object and gives a mask (used for copy-paste training images)."""

    def __init__(self):
        from ultralytics import SAM
        self.sam = SAM(str(ROOT / "models" / "mobile_sam.pt"))

    def refine(self, frame, box):
        """(box, mask) or None if SAM's answer does not look like the same object."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = 0.15 * max(x2 - x1, y2 - y1)   # let SAM see all of it even when the rough box was partial
        prompt = [max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)]
        r = self.sam(frame, bboxes=[prompt], verbose=False)[0]
        if r.masks is None:
            return None
        mask = r.masks.data[0].cpu().numpy().astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return None
        sb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        area = lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1]))     # noqa: E731
        inter = (max(0, min(x2, sb[2]) - max(x1, sb[0])) * max(0, min(y2, sb[3]) - max(y1, sb[1])))
        if inter / area(box) < 0.7 or area(sb) > 3 * area(box):   # not the same thing, or spilled onto the table
            return None
        return sb, mask


def save_label(d, stem, frame, box, mask=None, cls=0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cv2.imwrite(str(d / "images" / f"{stem}.jpg"), frame)
    (d / "labels" / f"{stem}.txt").write_text(
        f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    if mask is not None:
        (d / "masks").mkdir(exist_ok=True)
        cv2.imwrite(str(d / "masks" / f"{stem}.png"), mask * 255)


def read_box(d, stem, shape):
    h, w = shape[:2]
    cx, cy, bw, bh = (float(v) for v in (d / "labels" / f"{stem}.txt").read_text().split()[1:5])
    return (int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h))


def consistency_filter(d, stamp):
    """Drop labels whose size is far off the other rounds at the SAME pose (the object only turned,
    so its size there should be similar): catches slivers and boxes that grabbed a shadow."""
    by_pose = {}
    for lab in (d / "labels").glob(f"{stamp}_r*_p*.txt"):
        v = lab.read_text().split()
        by_pose.setdefault(lab.stem.split("_p")[-1], []).append((lab.stem, float(v[3]) * float(v[4])))
    dropped = []
    for items in by_pose.values():
        if len(items) < 3:
            continue
        med = float(np.median([a for _, a in items]))
        dropped += [stem for stem, a in items if not 0.4 * med <= a <= 2.5 * med]
    rej = d / "rejected"
    for stem in dropped:
        rej.mkdir(exist_ok=True)
        for sub, ext in (("images", ".jpg"), ("labels", ".txt"), ("masks", ".png")):
            f = d / sub / f"{stem}{ext}"
            if f.exists():
                shutil.move(str(f), str(rej / f.name))
    return dropped


def object_box(frame, background, min_frac=0.002, max_frac=0.4):
    """Bounding box of what differs from the empty-table picture, or None."""
    a = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    # Line the photo up with the empty-table photo first: the arm repeats to a few pixels, and
    # on a patterned table even that would show up as fake edges.
    (dx, dy), _ = cv2.phaseCorrelate(np.float32(b), np.float32(a))
    if 0.3 < np.hypot(dx, dy) < 25:
        a = cv2.warpAffine(a, np.float32([[1, 0, -dx], [0, 1, -dy]]), (a.shape[1], a.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    diff = cv2.absdiff(a, b)
    thr = max(25, int(np.percentile(diff, 99) * 0.5))
    mask = (diff > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)                 # remove texture speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)  # fill the object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    h, w = mask.shape
    area = cv2.contourArea(c) / (h * w)
    if not (min_frac <= area <= max_frac):
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    return x, y, x + bw, y + bh


def where_hint(box, shape, centred=False):
    """'' if the box is usable; otherwise what to do. centred=True (placing the object) also asks
    for it near the middle of the picture, so it stays in view when the arm shifts between poses."""
    if box is None:
        return "I cannot see it: put it nearer the middle of the work area, under the camera."
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    if centred:
        cx, cy = (x1 + x2) / 2 / w - 0.5, (y1 + y2) / 2 / h - 0.5
        off = [side for side, on in (("right", cx > 0.2), ("left", cx < -0.2),
                                     ("bottom", cy > 0.2), ("top", cy < -0.2)) if on]
        if off:
            return (f"Too far to the {' and '.join(off)} of the picture: move it toward the middle "
                    "(watch the live feed).")
    edge = []
    if x1 < 5: edge.append("left")             # noqa: E701
    if x2 > w - 5: edge.append("right")        # noqa: E701
    if y1 < 5: edge.append("top")              # noqa: E701
    if y2 > h - 5: edge.append("bottom")       # noqa: E701
    return f"It touches the {' and '.join(edge)} edge of the picture: move it toward the middle." if edge else ""


def cmd_add(name, rounds):
    from iro.motion import Arm
    from iro.pick import look_at, pick_config
    cfg = pick_config()
    name = name.strip().replace(" ", "_")
    d = OBJ_DIR / name
    if d.exists() and input(f"'{name}' exists. Add more pictures to it? [y/N] ").lower() != "y":
        return 1
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    (d / "background").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    refiner = Refiner()
    cap = open_camera()
    cap.status = f"teaching {name}: moving to the work area"
    kept = 0
    with Arm() as arm:
        poses = capture_poses(arm, cfg)
        print(f"{len(poses)} camera poses. Moving to the work area...")
        look_at(arm, cfg, cfg["look_xy"])
        cap.status = "STEP 1: clear the work area, then press Enter"
        input("\n1. CLEAR the work area under the camera (nothing there, hands away). Enter. ")
        bgs = []
        cap.lock_exposure()
        for i, (_, q) in enumerate(poses):
            cap.status = f"recording the empty table {i + 1}/{len(poses)}"
            arm.move_joints_repeatable(q)
            time.sleep(0.4)
            bg = grab(cap, n=3)
            bgs.append(bg)
            cv2.imwrite(str(d / "background" / f"{stamp}_p{i:02d}.jpg"), bg)
        # Repeatability check: back to two poses; the empty table must still match its picture,
        # or every label would be garbage (arm not returning to the same pose, or light changing)
        for i in (0, len(poses) // 2):
            arm.move_joints_repeatable(poses[-1 - i][1])
            arm.move_joints_repeatable(poses[i][1])
            time.sleep(0.4)
            if object_box(grab(cap, n=3), bgs[i]) is not None:
                cap.status = "STOPPED: empty table changed (light or arm) - see terminal"
                print("STOPPED: the empty table no longer matches its picture at the same pose. Either the light "
                      "changed (shadows, a moving person, sunlight) or the arm does not return to exactly the same "
                      "pose. Steady the light and try again.")
                cap.release()
                return 1
        print("   empty table recorded; poses repeat exactly.")
        centre = int(np.argmin([np.linalg.norm(np.subtract(p, [*cfg["look_xy"], cfg["table_z"] + 0.14])) for p, _ in poses]))
        arm.move_joints_repeatable(poses[centre][1])

        def placement(f):   # live: box around whatever differs from the empty table
            box = object_box(f, bgs[centre])
            hint = where_hint(box, f.shape, centred=True)
            return draw_box(f, box, (0, 255, 0) if box is not None and not hint else (0, 0, 255)), \
                ("ok, press Enter" if box is not None and not hint else hint.split(":")[0])
        cap.overlay = placement
        cap.status = f"STEP 2: put the {name} in view (green box = good)"
        while True:
            input(f"\n2. Put the {name} in the middle of the work area, under the camera. Hands away. Enter. ")
            time.sleep(0.4)
            box = object_box(grab(cap, n=3), bgs[centre])
            hint = where_hint(box, bgs[centre].shape, centred=True)
            if not hint:
                print("   I can see it.")
                break
            print("   " + hint)
        cap.overlay = None
        for r in range(rounds):
            if r:
                cap.status = f"round {r + 1}/{rounds}: turn the {name} a quarter turn, then Enter"
                input(f"\n   Round {r + 1}/{rounds}: turn the {name} a quarter turn (or onto another side), "
                      "same spot, hands away. Enter. ")
            got = 0
            arm.move_joints_repeatable(poses[centre][1])   # every round starts from the same pose
            for i, (_, q) in enumerate(poses):
                arm.move_joints_repeatable(q)
                time.sleep(0.4)
                cap.status = f"round {r + 1}/{rounds}, picture {i + 1}/{len(poses)}"
                f = grab(cap)
                box = object_box(f, bgs[i])
                if box is None or where_hint(box, f.shape):
                    cap.flash(draw_box(f, box, (0, 0, 255)), f"round {r + 1} picture {i + 1}: object not in view")
                    continue
                refined = refiner.refine(f, box)
                if refined is None or where_hint(refined[0], f.shape):
                    cap.flash(draw_box(f, box, (0, 0, 255)), f"round {r + 1} picture {i + 1}: unsure of the outline")
                    continue
                sbox, mask = refined
                shown = draw_box(f, sbox, (0, 200, 255), name)
                shown[mask > 0] = (shown[mask > 0] * 0.6 + np.array([0, 120, 0]) * 0.4).astype(np.uint8)
                cap.flash(shown, f"round {r + 1} picture {i + 1}: labelled")
                save_label(d, f"{stamp}_r{r}_p{i:02d}", f, sbox, mask)
                got += 1
            kept += got
            print(f"   round {r + 1}: {got}/{len(poses)} pictures labelled")
        look_at(arm, cfg, cfg["look_xy"])
    cap.release()
    dropped = consistency_filter(d, stamp)
    if dropped:
        print(f"   {len(dropped)} labels looked wrong next to the other rounds (moved to objects/{name}/rejected/)")
    make_sheet(name)
    print(f"\nSaved {kept - len(dropped)} labelled pictures of '{name}' in objects/{name}/.")
    print(f"CHECK the boxes: {SHEET.as_posix().format(name=name)} (each box should hug the object).")
    print("Then: iro object add <another> ... and finally: iro object train")
    return 0


def make_sheet(name, cols=6, size=(213, 160)):
    d = OBJ_DIR / name
    tiles = []
    for img in sorted((d / "images").glob("*.jpg"))[-36:]:
        f = cv2.imread(str(img))
        lab = (d / "labels" / f"{img.stem}.txt").read_text().split()
        h, w = f.shape[:2]
        cx, cy, bw, bh = (float(v) for v in lab[1:5])
        cv2.rectangle(f, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)),
                      (0, 200, 255), 3)
        tiles.append(cv2.resize(f, size))
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(SHEET).format(name=name), np.vstack(rows))


# --- training ----------------------------------------------------------------

def taught():
    return sorted(p.name for p in OBJ_DIR.iterdir() if (p / "images").is_dir()) if OBJ_DIR.exists() else []


def cmd_list():
    names = taught()
    if not names:
        print("No objects taught yet: iro object add <name>")
        return 0
    trained = json.loads((OBJ_DIR / "model" / "names.json").read_text()) if (OBJ_DIR / "model" / "names.json").exists() else []
    for n in names:
        k = len(list((OBJ_DIR / n / "images").glob("*.jpg")))
        print(f"  {n:20} {k:4} pictures   {'in the model' if n in trained else 'NOT trained yet'}")
    return 0


def cmd_relabel(name):
    """Re-do the boxes of already-captured pictures with SAM (and save masks)."""
    d = OBJ_DIR / name
    refiner = Refiner()
    ok = bad = 0
    for img in sorted((d / "images").glob("*.jpg")):
        f = cv2.imread(str(img))
        refined = refiner.refine(f, read_box(d, img.stem, f.shape))
        if refined is None or where_hint(refined[0], f.shape):
            bad += 1
            (d / "rejected").mkdir(exist_ok=True)
            for sub, ext in (("images", ".jpg"), ("labels", ".txt"), ("masks", ".png")):
                g = d / sub / f"{img.stem}{ext}"
                if g.exists():
                    shutil.move(str(g), str(d / "rejected" / g.name))
            continue
        save_label(d, img.stem, f, *refined)
        ok += 1
    stamps = {p.stem.split("_r")[0] for p in (d / "labels").glob("*.txt")}
    dropped = sum(len(consistency_filter(d, st)) for st in stamps)
    make_sheet(name)
    print(f"{name}: {ok - dropped} pictures relabelled with SAM, {bad + dropped} rejected (in objects/{name}/rejected/). "
          f"Check: objects/{name}/check.jpg")
    return 0


def synthesize(names, out_img, out_lab, per_object, rng):
    """Copy-paste training pictures: cut each object out with its SAM mask and paste it, at random
    size, angle and place (1-3 objects per picture), onto the empty-table photos."""
    bgs = [p for n in names for p in (OBJ_DIR / n / "background").glob("*.jpg")]
    cuts = []
    for cls, n in enumerate(names):
        for m in sorted((OBJ_DIR / n / "masks").glob("*.png")):
            img = cv2.imread(str(OBJ_DIR / n / "images" / f"{m.stem}.jpg"))
            mask = cv2.imread(str(m), 0)
            if img is None or mask is None:
                continue
            ys, xs = np.nonzero(mask)
            y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            cuts.append((cls, img[y1:y2, x1:x2], mask[y1:y2, x1:x2]))
    if not bgs or not cuts:
        return 0
    made = 0
    for i in range(per_object * len(names)):
        bg = cv2.imread(str(bgs[rng.integers(len(bgs))]))
        h, w = bg.shape[:2]
        lines, placed = [], []
        for _ in range(int(rng.integers(1, 4))):
            cls, obj, m = cuts[rng.integers(len(cuts))]
            scale = float(np.sqrt(rng.uniform(0.01, 0.12) * h * w / (obj.shape[0] * obj.shape[1])))
            angle = float(rng.uniform(0, 360))
            oh, ow = obj.shape[:2]
            M = cv2.getRotationMatrix2D((ow / 2, oh / 2), angle, scale)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(oh * sin + ow * cos) + 1, int(oh * cos + ow * sin) + 1
            if nw >= w - 4 or nh >= h - 4:
                continue
            M[0, 2] += nw / 2 - ow / 2
            M[1, 2] += nh / 2 - oh / 2
            o = cv2.warpAffine(obj, M, (nw, nh))
            a = cv2.warpAffine(m, M, (nw, nh)).astype(np.float32) / 255
            a = cv2.GaussianBlur(a, (5, 5), 0)[..., None]          # soft edge, no cut-out halo
            x, y = int(rng.integers(0, w - nw)), int(rng.integers(0, h - nh))
            if any(not (x + nw < px1 or px2 < x or y + nh < py1 or py2 < y) for px1, py1, px2, py2 in placed):
                continue                                            # no stacking: keeps every label honest
            placed.append((x, y, x + nw, y + nh))
            bright = rng.uniform(0.85, 1.05)                        # lighting differences (never brighter-than-white)
            region = bg[y:y + nh, x:x + nw].astype(np.float32)
            bg[y:y + nh, x:x + nw] = np.clip(region * (1 - a) + o.astype(np.float32) * bright * a, 0, 255).astype(np.uint8)
            ys, xs = np.nonzero(a[..., 0] > 0.5)
            if len(xs):
                bx1, by1, bx2, by2 = x + xs.min(), y + ys.min(), x + xs.max(), y + ys.max()
                lines.append(f"{cls} {(bx1 + bx2) / 2 / w:.6f} {(by1 + by2) / 2 / h:.6f} {(bx2 - bx1) / w:.6f} {(by2 - by1) / h:.6f}")
        if lines:
            cv2.imwrite(str(out_img / f"synth_{i:05d}.jpg"), bg)
            (out_lab / f"synth_{i:05d}.txt").write_text("\n".join(lines) + "\n")
            made += 1
    return made


def cmd_train(epochs):
    from ultralytics import YOLO
    names = taught()
    if not names:
        print("Nothing to train: iro object add <name> first")
        return 1
    work = ROOT / "datasets" / "objects"
    shutil.rmtree(work, ignore_errors=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (work / split / "images").mkdir(parents=True)
        (work / split / "labels").mkdir(parents=True)
    count, n_train = {}, 0
    for cls, n in enumerate(names):
        imgs = sorted((OBJ_DIR / n / "images").glob("*.jpg"))
        count[n] = len(imgs)
        # Hold out whole ROUNDS for validation (the object in an orientation never trained on):
        # random frames would put near-identical neighbours in both sets and flatter the score.
        rounds = sorted({p.stem.rsplit("_p", 1)[0] for p in imgs})
        held = set(rounds[-1:]) if len(rounds) > 1 else set()
        for img in imgs:
            split = "val" if img.stem.rsplit("_p", 1)[0] in held or (not held and rng.random() < 0.15) else "train"
            n_train += split == "train"
            shutil.copy(img, work / split / "images" / img.name)
            lab = (OBJ_DIR / n / "labels" / f"{img.stem}.txt").read_text().split()
            (work / split / "labels" / f"{img.stem}.txt").write_text(" ".join([str(cls)] + lab[1:]) + "\n")
    synth = synthesize(names, work / "train" / "images", work / "train" / "labels", per_object=150, rng=rng)
    # Empty-table pictures teach "this is not an object"; about 10% of the training set is right
    bgs = [p for n in names for p in sorted((OBJ_DIR / n / "background").glob("*.jpg"))]
    for bg in [bgs[i] for i in rng.permutation(len(bgs))[: max(1, (n_train + synth) // 10)]] if bgs else []:
        shutil.copy(bg, work / "train" / "images" / f"bg_{bg.parent.parent.name}_{bg.name}")
        (work / "train" / "labels" / f"bg_{bg.parent.parent.name}_{bg.stem}.txt").write_text("")
    (work / "data.yaml").write_text(f"path: {work}\ntrain: train/images\nval: val/images\nnames: {json.dumps(names)}\n")
    print(f"Training on {count} real pictures + {synth} copy-paste pictures. About 15-30 minutes; "
          "the robot can't use the GPU meanwhile.")
    model = YOLO(str(STOCK_MODEL))
    model.train(data=str(work / "data.yaml"), epochs=epochs, imgsz=640, batch=8, workers=2, device=0,
                patience=20, project=str(work / "runs"), name="train", exist_ok=True, verbose=False, plots=False,
                # black-and-white camera: no colour jitter, but light changes; objects at any angle/size
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, degrees=15, scale=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=10)
    best = work / "runs" / "train" / "weights" / "best.pt"
    OBJ_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OBJ_MODEL)
    (OBJ_DIR / "model" / "names.json").write_text(json.dumps(names))
    metrics = YOLO(str(OBJ_MODEL)).val(data=str(work / "data.yaml"), device=0, verbose=False, plots=False)
    print(f"\nDone: objects/model/objects.pt knows {names}. Score on the held-out turn of each object "
          f"(mAP50): {metrics.box.map50:.2f}  (1.0 = perfect, above 0.8 is good). Try it: iro vision")
    return 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if a[0] in ("show", "forget", "memory"):
        from iro.instant import main as instant_main
        sys.argv = [sys.argv[0]] + a
        return instant_main()
    if a[0] == "add" and len(a) >= 2:
        rounds = int(a[a.index("--rounds") + 1]) if "--rounds" in a else 4
        return cmd_add(a[1], rounds)
    if a[0] == "list":
        return cmd_list()
    if a[0] == "relabel" and len(a) == 2:
        return cmd_relabel(a[1])
    if a[0] == "train":
        return cmd_train(int(a[a.index("--epochs") + 1]) if "--epochs" in a else 60)
    if a[0] == "remove" and len(a) == 2:
        if input(f"Delete all pictures of '{a[1]}'? Type yes: ") == "yes":
            shutil.rmtree(OBJ_DIR / a[1])
            print("Removed. Run `iro object train` to update the model.")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
