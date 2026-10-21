"""Lock-on: teach the robot one object from 10-20 annotated views, then find it anywhere.

    iro track teach NAME       guided tour: you hold/place the object as told, freeze a frame
                               (Enter), DRAG a box around it on the live page; SAM outlines it
    iro track NAME             follow it (see iro/track.py)

Finding it uses the Matcher idea (DINOv2 patch matching, one-shot, no training): the object is
a bank of DINOv2 patch features from INSIDE its outlines, the room is a bank from outside; every
patch of a new picture is scored "closer to the object than to the room", and the best blob is
the object. Tested on held-out Tapak Sirih photos from just 7 views: 14/20 found, median IoU 0.66,
62 ms per picture. Room patches that come close are added as hard negatives while tracking.

The annotated pictures are also saved in objects/NAME/ (images, labels, masks), so
`iro object train` can turn them into a YOLO model later.
"""

import sys
import time

import cv2
import numpy as np

from iro import ROOT

PATCH = 14
W, H = 448, 336                      # 32 x 24 patches for a 640x480 picture
BANKS = ROOT / "objects" / "memory"
TOUR = [
    "Hold it in front of the camera, front side facing it",
    "Turn it about 45 degrees",
    "Turn it 90 degrees (show its side)",
    "Show the back",
    "Show the top",
    "Tilt it toward the camera",
    "Hold it CLOSER to the camera",
    "Hold it FURTHER away",
    "Move it to the LEFT side of the picture",
    "Move it to the RIGHT side of the picture",
    "Put it on the floor, in view",
    "On the floor, turned another way",
]


class PatchMatcher:
    def __init__(self):
        import timm
        import torch
        from iro.instant import DINO_FILE
        self.torch = torch
        net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0,
                                img_size=224, dynamic_img_size=True)
        net.load_state_dict(torch.load(DINO_FILE, map_location="cpu"))
        self.net = net.eval().cuda().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        self.fg = torch.zeros((0, 384), device="cuda")
        self.bg = torch.zeros((0, 384), device="cuda")

    def patches(self, img):
        torch = self.torch
        x = torch.from_numpy(cv2.resize(img, (W, H))[:, :, ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
        with torch.no_grad():
            t = self.net.forward_features(((x - self.mean) / self.std).half()).float()
        t = t[:, self.net.num_prefix_tokens:, :].reshape(H // PATCH, W // PATCH, -1)
        return torch.nn.functional.normalize(t, dim=-1)

    @staticmethod
    def small(mask):
        return cv2.resize(mask.astype(np.uint8), (W // PATCH, H // PATCH), interpolation=cv2.INTER_AREA) > 0

    def add_view(self, img, mask):
        """Learn the object from a picture + its outline (and the room around it)."""
        t = self.patches(img)
        s = self.torch.from_numpy(self.small(mask > 0)).cuda()
        if s.sum() == 0:
            return
        self.fg = self.torch.cat([self.fg, t[s]])[-4000:]
        self.bg = self.torch.cat([self.bg, t[~s][::2]])[-12000:]

    def score_map(self, img):
        t = self.patches(img)
        k = 3
        sf = (t @ self.fg.T).topk(min(k, len(self.fg)), dim=-1).values.mean(-1)
        sb = (t @ self.bg.T).topk(min(k, len(self.bg)), dim=-1).values.mean(-1) if len(self.bg) else 0 * sf
        return (sf - sb).cpu().numpy(), t

    def locate(self, img, near=None, radius=None, thr=0.0):
        """(box in pixels, strength) of the object, preferring a blob near `near`; None if absent."""
        s, t = self.score_map(img)
        n, lab, st, _ = cv2.connectedComponentsWithStats((s > thr).astype(np.uint8))
        if n <= 1:
            return None
        h, w = img.shape[:2]
        sx, sy = w / (W // PATCH), h / (H // PATCH)
        best = None
        for i in range(1, n):
            x, y, bw, bh = st[i, :4]
            box = np.array([x * sx, y * sy, (x + bw) * sx, (y + bh) * sy])
            strength = float(s[lab == i].sum())
            if near is not None and radius is not None:
                c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                if np.linalg.norm(c - near) > radius:
                    continue
            if best is None or strength > best[1]:
                best = (box, strength, i)
        if best is None:
            return None
        # hard negatives: strong room patches outside the chosen blob teach "not this"
        outside = (lab != best[2]) & (s > thr)
        if outside.any():
            self.bg = self.torch.cat([self.bg, t[self.torch.from_numpy(outside).cuda()]])[-12000:]
        return best[0], best[1]

    def save(self, name):
        BANKS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BANKS / f"{name}.patches.npz", fg=self.fg.cpu().numpy(), bg=self.bg.cpu().numpy())

    def load(self, name):
        f = BANKS / f"{name}.patches.npz"
        if not f.exists():
            return False
        d = np.load(f)
        self.fg = self.torch.from_numpy(d["fg"]).cuda()
        self.bg = self.torch.from_numpy(d["bg"]).cuda()
        return True


def sam_outline(refiner, img, box):
    """SAM outline from a drawn box (falls back to the box itself)."""
    r = refiner.sam(img, bboxes=[list(box)], verbose=False)[0]
    if r.masks is None:
        m = np.zeros(img.shape[:2], np.uint8)
        x1, y1, x2, y2 = map(int, box)
        m[y1:y2, x1:x2] = 1
        return m
    return r.masks.data[0].cpu().numpy().astype(np.uint8)


def cmd_teach(name, views=len(TOUR)):
    """Guided tour: freeze, draw a box on the live page, SAM outlines it, learn it."""
    from iro.instant import Recognizer, crop
    from iro.objects import OBJ_DIR, Refiner, open_camera, save_label
    name = name.strip().replace(" ", "_")
    refiner, pm, rec = Refiner(), PatchMatcher(), Recognizer()
    pm.load(name)                                     # add to what it already knows
    cam = open_camera()
    d = OBJ_DIR / name
    for sub in ("images", "labels", "masks"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Teaching '{name}'. Open http://localhost:8080. For each step: do what it says, press Enter "
          "to FREEZE the picture, then DRAG a box around the object on the page.  'f' + Enter = finish.")
    done = 0
    try:
        for i, step in enumerate(TOUR[:views]):
            cam.status = f"{i + 1}/{views}: {step} - then press Enter"
            a = input(f"\n{i + 1}/{views}: {step}. Enter to freeze ('f' = finish): ").strip().lower()
            if a == "f":
                break
            f = cam.latest()
            cam.flash(f, "FROZEN: drag a box around the object", 600)
            box = cam.feed.ask_box(f.shape, timeout=120)
            if box is None:
                print("  no box drawn; skipping this view")
                cam.hold_until = 0
                continue
            mask = sam_outline(refiner, f, box)
            ys, xs = np.nonzero(mask)
            tight = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else box
            shown = f.copy()
            shown[mask > 0] = (shown[mask > 0] * 0.5 + np.array([0, 160, 0]) * 0.5).astype(np.uint8)
            cam.flash(shown, "outline - keep it? (terminal)", 600)
            if input("  Is the green outline the object? Enter = yes, n = redo later: ").strip().lower() == "n":
                cam.hold_until = 0
                continue
            cam.hold_until = 0
            pm.add_view(f, mask)
            save_label(d, f"{stamp}_ann_p{i:02d}", f, tight, mask)
            rec.memory.add(name, rec.fp([crop(f, tight)]))
            done += 1
            print(f"  learnt view {done}")
    except KeyboardInterrupt:
        print()
    cam.release()
    if done:
        pm.save(name)
        print(f"\nLearnt {done} views of '{name}' ({len(pm.fg)} object patches). Try: iro track {name}")
    return 0 if done else 1


def flow_jacobian(arm, cam, q0, joints=(0, 3), step_deg=6):
    """How the WHOLE picture shifts per radian of each joint (2 x len(joints)), from small nudges.
    Uses the static room (phase correlation), so the object does not need to stay still."""
    J = np.zeros((2, len(joints)))
    for k, j in enumerate(joints):
        shots = []
        for sign in (+1, -1):
            q = q0.copy()
            q[j] += sign * np.radians(step_deg)
            arm.move_joints(q, duration=0.6)
            time.sleep(0.4)
            shots.append(np.float32(cv2.cvtColor(cam.latest(), cv2.COLOR_BGR2GRAY)))
        (dx, dy), resp = cv2.phaseCorrelate(shots[1], shots[0])
        J[:, k] = np.array([dx, dy]) / np.radians(2 * step_deg)
        arm.move_joints(q0, duration=0.6)
    return J


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "teach":
        return cmd_teach(a[1])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
