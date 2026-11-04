"""Instant object recognition: show an object once, name it, and the robot knows it. No training.

    iro object show Tapak_Sirih     hold/put it in the middle of the camera view for ~15 s
    iro object forget Tapak_Sirih
    iro object memory               rebuild the memory from pictures already taken (iro object add)

How it works (the "segment everything + DINOv2 matching" approach from instance-detection
research, which beat trained detectors on few-example robot tasks):
  1. YOLOE in prompt-free mode finds every thing in the picture, whatever it is
  2. DINOv2 turns each thing into a fingerprint (384 numbers describing its look)
  3. a thing gets a name if its fingerprint is clearly closer to that object's saved
     fingerprints than to the "not an object" fingerprints (table, floor, furniture)
Tested 20 Oct on held-out pictures of the Tapak Sirih: found 12/12, no false alarms in 306.
"""

import sys
import time

import cv2
import numpy as np

from iro import ROOT

MEM = ROOT / "objects" / "memory"            # <name>.npy fingerprints; _background.npy = "not an object"
MODELS = ROOT / "models"
FINDER = MODELS / "yoloe-11s-seg-pf.pt"
DINO_FILE = MODELS / "dinov2_vits14.pth"      # kept locally so it works offline at the competition
MARGIN = -0.10   # name a thing if (closest object fingerprint) - (closest background one) > MARGIN


class Fingerprinter:
    """DINOv2-small, 224 px input, half precision on the GPU: ~4 ms per object."""

    def __init__(self):
        import timm
        import torch
        self.torch = torch
        name = "vit_small_patch14_dinov2.lvd142m"
        if DINO_FILE.exists():
            self.net = timm.create_model(name, pretrained=False, num_classes=0, img_size=224)
            self.net.load_state_dict(torch.load(DINO_FILE, map_location="cpu"))
        else:
            self.net = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
            torch.save(self.net.state_dict(), DINO_FILE)
        self.net = self.net.eval().cuda().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def __call__(self, crops):
        torch = self.torch
        if not crops:
            return np.zeros((0, 384), np.float32)
        out = []
        for i in range(0, len(crops), 32):
            x = torch.stack([torch.from_numpy(cv2.resize(c, (224, 224))[:, :, ::-1].copy()).permute(2, 0, 1)
                             for c in crops[i:i + 32]]).cuda().float() / 255
            with torch.no_grad():
                e = self.net(((x - self.mean) / self.std).half()).float()
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.vstack(out)


def crop(img, box, pad=0.1):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    p = pad * max(x2 - x1, y2 - y1)
    return img[int(max(0, y1 - p)):int(min(h, y2 + p)), int(max(0, x1 - p)):int(min(w, x2 + p))]


class Memory:
    def __init__(self):
        MEM.mkdir(parents=True, exist_ok=True)
        self.objects = {p.stem: np.load(p) for p in MEM.glob("*.npy") if not p.stem.startswith("_")}
        bg = MEM / "_background.npy"
        self.background = np.load(bg) if bg.exists() else np.zeros((0, 384), np.float32)

    def add(self, name, fps):
        cur = self.objects.get(name, np.zeros((0, 384), np.float32))
        self.objects[name] = np.vstack([cur, fps])[-300:]          # keep the newest 300 per object
        np.save(MEM / f"{name}.npy", self.objects[name])

    def add_background(self, fps):
        self.background = np.vstack([self.background, fps])[-3000:]
        np.save(MEM / "_background.npy", self.background)

    def forget(self, name):
        self.objects.pop(name, None)
        (MEM / f"{name}.npy").unlink(missing_ok=True)

    def match(self, fps):
        """[(name or None, margin)] per fingerprint."""
        if not self.objects or not len(fps):
            return [(None, -1.0)] * len(fps)
        neg = (fps @ self.background.T).max(1) if len(self.background) else np.zeros(len(fps))
        names = list(self.objects)
        sims = np.stack([(fps @ self.objects[n].T).max(1) for n in names], 1)
        best = sims.argmax(1)
        margin = sims[np.arange(len(fps)), best] - neg
        return [(names[b] if m > MARGIN else None, float(m)) for b, m in zip(best, margin)]


class Recognizer:
    """Find every thing (YOLOE prompt-free), then name the ones it remembers."""

    def __init__(self):
        from ultralytics import YOLOE
        self.finder = YOLOE(str(FINDER))
        self.fp = Fingerprinter()
        self.memory = Memory()

    def things(self, frame, conf=0.1, min_frac=0.003, max_frac=0.5):
        r = self.finder.predict(frame, device=0, verbose=False, conf=conf)[0]
        h, w = frame.shape[:2]
        out = []
        for b in r.boxes.xyxy.cpu().numpy():
            a = (b[2] - b[0]) * (b[3] - b[1]) / (h * w)
            if min_frac <= a <= max_frac:
                out.append(b)
        return out

    def recognize(self, frame):
        """[(name, margin, (u, v) centre, box)] for every remembered object in view, best first."""
        boxes = self.things(frame)
        if not boxes or not self.memory.objects:
            return []
        found = []
        for b, (name, m) in zip(boxes, self.memory.match(self.fp([crop(frame, b) for b in boxes]))):
            if name:
                found.append((name, m, np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), b))
        found.sort(key=lambda d: -d[1])
        keep = []   # one box per object: the same thing is often found twice (whole + part)
        for d in found:
            if all(not _overlap(d[3], k[3]) for k in keep if k[0] == d[0]):
                keep.append(d)
        return keep


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > 0.3 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def centre_thing(boxes, shape):
    """The smallest found thing that covers the middle of the picture (what you are showing)."""
    h, w = shape[:2]
    inside = [b for b in boxes if b[0] < w / 2 < b[2] and b[1] < h / 2 < b[3]]
    return min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if inside else None


class OakSource:
    """Lets `show` teach from the OAK-D colour camera (what EFAD picks with) instead of the wrist."""

    def __init__(self):
        from iro.oak import Oak
        from iro.stream import feed
        self.oak, self.feed, self.status, self.hold_until = Oak(), feed(), "", 0

    def read(self):
        return True, self.oak.read(1)[0]

    def flash(self, img, text, seconds=0.3):
        self.feed.show(img, text)

    def release(self):
        self.oak.close()


def cmd_show(name, seconds=15, oak=False):
    from iro.objects import draw_box, open_camera
    name = name.strip().replace(" ", "_")
    rec = Recognizer()
    cam = OakSource() if oak else open_camera()
    print(f"Show the {name}: put it in the MIDDLE of the camera view (live feed), then slowly turn and "
          f"move it for {seconds} s (different sides, nearer and further). Ctrl+C to stop early.")
    cam.status = f"show the {name} in the middle of the picture"
    input("Ready? Enter to start. ")
    got, bgn, t0 = 0, 0, time.time()
    try:
        while time.time() - t0 < seconds:
            ok, f = cam.read()
            if not ok:
                continue
            boxes = rec.things(f)
            target = centre_thing(boxes, f.shape)
            if target is None:
                cam.flash(f, "nothing in the middle of the picture: move it to the centre", 0.3)
                continue
            others = [b for b in boxes if not _overlap(b, target)]
            fps = rec.fp([crop(f, target)] + [crop(f, b) for b in others])
            rec.memory.add(name, fps[:1])
            if len(others):
                rec.memory.add_background(fps[1:])
            got, bgn = got + 1, bgn + len(others)
            cam.flash(draw_box(f, target, (0, 200, 255), name),
                      f"remembering {name}: {got} views ({int(seconds - (time.time() - t0))} s left)", 0.2)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    cam.release()
    print(f"Remembered {got} views of {name} (+{bgn} 'not this' views). Try it now: iro vision")
    return 0 if got else 1


def cmd_memory():
    """Build fingerprints from pictures taken with `iro object add` (boxes + empty-table photos)."""
    from iro.objects import OBJ_DIR, read_box
    rec = Recognizer()
    for d in sorted(p for p in OBJ_DIR.iterdir() if (p / "images").is_dir()):
        crops = []
        for img in sorted((d / "images").glob("*.jpg")):
            f = cv2.imread(str(img))
            crops.append(crop(f, read_box(d, img.stem, f.shape)))
        if crops:
            rec.memory.add(d.name, rec.fp(crops))
        negs = []
        for bg in sorted((d / "background").glob("*.jpg")):
            f = cv2.imread(str(bg))
            negs += [crop(f, b) for b in rec.things(f)]
        if negs:
            rec.memory.add_background(rec.fp(negs))
        print(f"  {d.name}: {len(crops)} views, {len(negs)} 'not an object' views")
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "show" and len(a) >= 2:
        return cmd_show(a[1], int(a[a.index("--seconds") + 1]) if "--seconds" in a else 15)
    if a and a[0] == "forget" and len(a) == 2:
        Memory().forget(a[1])
        print(f"Forgot {a[1]}.")
        return 0
    if a and a[0] == "memory":
        return cmd_memory()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
