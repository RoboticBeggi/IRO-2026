"""Teach the robot your own objects: the arm films them, labels them, trains YOLO.

    iro object add wayang      record a new object called "wayang" (arm does the angles)
    iro object list            taught objects and how many pictures each has
    iro object train           train YOLO on all taught objects (~10-20 min on the Jetson)
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


def open_camera():
    dev = load_config()["devices"]["arm_camera"]["device"]
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open the wrist camera {dev} (is `iro vision` or `iro up camera` using it?)")
    return cap


def grab(cap, n=1):
    """Fresh frame(s): drop the ones buffered while the arm moved; median of n reduces noise."""
    for _ in range(4):
        cap.grab()
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            frames.append(f)
    if not frames:
        raise RuntimeError("wrist camera gives no image")
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if n > 1 else frames[0]


class Detector:
    """Stock YOLO (80 everyday classes) + the model trained on your own objects, together."""

    def __init__(self, conf=0.4):
        from ultralytics import YOLO
        self.conf = conf
        self.models = [("stock", YOLO(str(STOCK_MODEL)))]
        if OBJ_MODEL.exists():
            self.models.append(("taught", YOLO(str(OBJ_MODEL))))

    def describe(self):
        out = ["stock YOLO (80 classes)"]
        if len(self.models) > 1:
            out.append("taught: " + ", ".join(self.models[1][1].names.values()))
        return out

    @property
    def taught_names(self):
        return list(self.models[1][1].names.values()) if len(self.models) > 1 else []

    def detect(self, frame, conf=None, draw=False):
        """[(name, conf, (u, v) centre, box xyxy, source)], best first; plus an annotated copy if draw."""
        found = []
        for source, m in self.models:
            r = m.predict(frame, device=0, verbose=False, conf=conf or self.conf)[0]
            for b, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls, r.boxes.conf):
                found.append((r.names[int(c)], float(p), np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), b, source))
        found.sort(key=lambda d: -d[1])
        if not draw:
            return found, None
        img = frame.copy()
        for name, p, _, b, source in found:
            color = (0, 200, 255) if source == "taught" else (0, 255, 0)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} {p:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return found, img


# --- capture -----------------------------------------------------------------

def capture_poses(arm, cfg):
    """Camera poses around the look pose that the arm can actually reach (same list every time)."""
    from iro.motion import SEEDS
    lx, ly = cfg["look_xy"]
    poses = []
    for dz in (0.0, 0.02):
        for dx in (-0.03, 0.0, 0.03):
            for dy in (-0.04, 0.0, 0.04):
                xyz = [lx + dx, ly + dy, cfg["look_z"] + dz]
                q, err = min((arm.kin.ik_down(xyz, s, tilt=np.radians(cfg["look_tilt_deg"])) for s in SEEDS),
                             key=lambda r: r[1])
                if err < 0.004:
                    poses.append(xyz)
    return poses


def object_box(frame, background, min_frac=0.002, max_frac=0.4):
    """Bounding box of what differs from the empty-table picture, or None."""
    a = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), (7, 7), 0)
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


def where_hint(box, shape):
    if box is None:
        return "I cannot see it: put it nearer the middle of the work area, under the camera."
    h, w = shape[:2]
    x1, y1, x2, y2 = box
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
    cap = open_camera()
    kept = 0
    with Arm() as arm:
        poses = capture_poses(arm, cfg)
        print(f"{len(poses)} camera poses. Moving to the work area...")
        arm.gripper(open=True)
        look_at(arm, cfg, cfg["look_xy"])
        input("\n1. CLEAR the work area under the camera (nothing there, hands away). Enter. ")
        bgs = []
        for i, xyz in enumerate(poses):
            arm.move_tip(xyz, tilt=np.radians(cfg["look_tilt_deg"]))
            time.sleep(0.4)
            bg = grab(cap, n=3)
            bgs.append(bg)
            cv2.imwrite(str(d / "background" / f"{stamp}_p{i:02d}.jpg"), bg)
        # Repeatability check: back to two poses; the empty table must still match its picture,
        # or every label would be garbage (arm not returning to the same pose, or light changing)
        for i in (0, len(poses) // 2):
            arm.move_tip(poses[-1 - i], tilt=np.radians(cfg["look_tilt_deg"]))
            arm.move_tip(poses[i], tilt=np.radians(cfg["look_tilt_deg"]))
            time.sleep(0.4)
            if object_box(grab(cap, n=3), bgs[i]) is not None:
                print("STOPPED: the empty table no longer matches its picture at the same pose. Either the light "
                      "changed (shadows, a moving person, sunlight) or the arm does not return to exactly the same "
                      "pose. Steady the light and try again.")
                cap.release()
                return 1
        print("   empty table recorded; poses repeat exactly.")
        centre = int(np.argmin([np.linalg.norm(np.subtract(p, [*cfg["look_xy"], cfg["look_z"]])) for p in poses]))
        arm.move_tip(poses[centre], tilt=np.radians(cfg["look_tilt_deg"]))
        while True:
            input(f"\n2. Put the {name} in the middle of the work area, under the camera. Hands away. Enter. ")
            time.sleep(0.4)
            box = object_box(grab(cap, n=3), bgs[centre])
            hint = where_hint(box, bgs[centre].shape)
            if not hint:
                print("   I can see it.")
                break
            print("   " + hint)
        for r in range(rounds):
            if r:
                input(f"\n   Round {r + 1}/{rounds}: turn the {name} a quarter turn (or onto another side), "
                      "same spot, hands away. Enter. ")
            got = 0
            for i, xyz in enumerate(poses):
                arm.move_tip(xyz, tilt=np.radians(cfg["look_tilt_deg"]))
                time.sleep(0.4)
                f = grab(cap)
                box = object_box(f, bgs[i])
                if box is None or where_hint(box, f.shape):
                    continue
                stem = f"{stamp}_r{r}_p{i:02d}"
                cv2.imwrite(str(d / "images" / f"{stem}.jpg"), f)
                h, w = f.shape[:2]
                x1, y1, x2, y2 = box
                (d / "labels" / f"{stem}.txt").write_text(
                    f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
                got += 1
            kept += got
            print(f"   round {r + 1}: {got}/{len(poses)} pictures labelled")
        look_at(arm, cfg, cfg["look_xy"])
    cap.release()
    make_sheet(name)
    print(f"\nSaved {kept} labelled pictures of '{name}' in objects/{name}/.")
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
    count = {}
    for cls, n in enumerate(names):
        imgs = sorted((OBJ_DIR / n / "images").glob("*.jpg"))
        count[n] = len(imgs)
        for img in imgs:
            split = "val" if rng.random() < 0.15 else "train"
            shutil.copy(img, work / split / "images" / img.name)
            lab = (OBJ_DIR / n / "labels" / f"{img.stem}.txt").read_text().split()
            (work / split / "labels" / f"{img.stem}.txt").write_text(" ".join([str(cls)] + lab[1:]) + "\n")
        # empty-table pictures: teach it what is NOT an object
        for bg in sorted((OBJ_DIR / n / "background").glob("*.jpg")):
            shutil.copy(bg, work / "train" / "images" / f"bg_{n}_{bg.name}")
            (work / "train" / "labels" / f"bg_{n}_{bg.stem}.txt").write_text("")
    (work / "data.yaml").write_text(f"path: {work}\ntrain: train/images\nval: val/images\nnames: {json.dumps(names)}\n")
    print(f"Training on {count} (+ empty-table pictures). About 10-20 minutes; the robot can't use the GPU meanwhile.")
    model = YOLO(str(STOCK_MODEL))
    model.train(data=str(work / "data.yaml"), epochs=epochs, imgsz=640, batch=8, workers=2, device=0,
                patience=15, project=str(work / "runs"), name="train", exist_ok=True, verbose=False, plots=False)
    best = work / "runs" / "train" / "weights" / "best.pt"
    OBJ_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OBJ_MODEL)
    (OBJ_DIR / "model" / "names.json").write_text(json.dumps(names))
    metrics = YOLO(str(OBJ_MODEL)).val(data=str(work / "data.yaml"), device=0, verbose=False, plots=False)
    print(f"\nDone: objects/model/objects.pt knows {names}. Detection quality (mAP50) {metrics.box.map50:.2f} "
          "(1.0 = perfect, above 0.8 is good). Try it: iro vision")
    return 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if a[0] == "add" and len(a) >= 2:
        rounds = int(a[a.index("--rounds") + 1]) if "--rounds" in a else 4
        return cmd_add(a[1], rounds)
    if a[0] == "list":
        return cmd_list()
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
