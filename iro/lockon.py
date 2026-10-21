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
MIN_PEAK = 0.35        # x ref_peak. Tape held out: real 0.13-0.42 (floor = lowest), look-alikes up to
                       # 0.13: strength alone cannot tell them apart, the size/shape/brightness gates do
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
        self.ref_peak = 0.34         # typical peak of a real sighting (Tape, 21 Oct); set by reference()
        self.profile = None          # size/shape/brightness ranges, from the taught views
        self.why, self.last, self.best_label = "", None, None

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

    def locate(self, img, near=None, radius=None, thr=0.0, min_peak=None, window=None):
        """(box in pixels, strength, peak) of the object, preferring a blob near `near`; None if absent
        (self.why says why). A blob only counts if
          - its best patch scores >= min_peak (default: MIN_PEAK x the object's held-out typical score),
          - its size, shape and brightness fit what the object looked like while it was taught
            (self.profile): a dark shadow or a long strip under the sofa is never the white tape.
        window=(x1,y1,x2,y2): search only there, ZOOMED in (finer patches, like a tracker's search
        region): used while locked, the whole picture is searched when lost."""
        min_peak = self.ref_peak * MIN_PEAK if min_peak is None else min_peak
        h, w = img.shape[:2]
        ox = oy = 0
        view = img
        if window is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in window)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 > 32 and y2 - y1 > 24:
                ox, oy, view = x1, y1, img[y1:y2, x1:x2]
        s, t = self.score_map(view)
        self.last = (s, t, view, ox, oy)
        n, lab, st, _ = cv2.connectedComponentsWithStats((s > thr).astype(np.uint8))
        self.why = "nothing looks like it"
        if n <= 1:
            return None
        vh, vw = view.shape[:2]
        sx, sy = vw / (W // PATCH), vh / (H // PATCH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scene = max(1.0, float(np.median(gray)))
        best, why = None, []
        for i in range(1, n):
            x, y, bw, bh = st[i, :4]
            box = np.array([ox + x * sx, oy + y * sy, ox + (x + bw) * sx, oy + (y + bh) * sy])
            strength = float(s[lab == i].sum())
            peak = float(s[lab == i].max())
            if peak < min_peak:
                why.append(f"weak {peak:.2f}")
                continue
            if near is not None and radius is not None:
                c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                if np.linalg.norm(c - near) > radius:
                    why.append("far from the predicted spot")
                    continue
            if self.profile is not None:
                bad = self.misfit(box, gray, scene, cv2.resize((lab == i).astype(np.uint8), (vw, vh),
                                                              interpolation=cv2.INTER_NEAREST) > 0, ox, oy)
                if bad:
                    why.append(bad)
                    continue
            if best is None or strength > best[1]:
                best = (box, strength, i, peak)
        if best is None:
            self.why = ", ".join(why[:3]) or self.why
            return None
        self.why = ""
        self.best_label = (lab, best[2])
        return best[0], best[1], best[3]

    def misfit(self, box, gray, scene, blob, ox=0, oy=0):
        """Why this blob cannot be the object (size / shape / brightness vs. the taught views), or ''."""
        a_min, a_max, r_min, r_max, b_min, b_max = self.profile
        bw, bh = box[2] - box[0], box[3] - box[1]
        if not a_min <= bw * bh <= a_max:
            return f"size {bw:.0f}x{bh:.0f}"
        if not r_min <= bw / max(bh, 1) <= r_max:
            return f"shape {bw / max(bh, 1):.1f}:1"
        ys, xs = np.nonzero(blob)
        px = gray[ys + oy, xs + ox] if len(xs) else gray[int(box[1]):int(box[3]), int(box[0]):int(box[2])].ravel()
        br = float(np.percentile(px, 90)) / scene if len(px) else 0.0
        if not b_min <= br <= b_max:
            return f"{'too dark' if br < b_min else 'too bright'} ({br:.2f} x scene)"
        return ""

    def learn_profile(self, views):
        """Size, shape and brightness ranges from (picture, outline) pairs, with margins."""
        a, r, b = [], [], []
        for img, mask in views:
            ys, xs = np.nonzero(mask > 0)
            if len(xs) < 20:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            a.append(bw * bh)
            r.append(bw / bh)
            b.append(np.percentile(g[ys, xs], 75) / max(1.0, float(np.median(g))))
        if len(a) >= 3:
            self.profile = np.array([min(a) * 0.35, max(a) * 1.8, min(r) / 1.7, max(r) * 1.7,
                                     min(b) * 0.85, max(b) * 1.25])

    def learn_scene(self, img, box, mask=None):
        """At lock-on: everything outside the confirmed box is 'not it' (this room, this shadow),
        what is inside is 'it' as it looks right now (online discriminative update, as in DiMP)."""
        t = self.patches(img)
        h, w = img.shape[:2]
        gh, gw = H // PATCH, W // PATCH
        x1, y1, x2, y2 = box
        inside = np.zeros((gh, gw), bool)
        inside[int(y1 / h * gh):int(np.ceil(y2 / h * gh)), int(x1 / w * gw):int(np.ceil(x2 / w * gw))] = True
        around = cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        obj = self.small(mask > 0) if mask is not None else cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        tt = self.torch
        self.bg = tt.cat([self.bg, t[tt.from_numpy(~around).cuda()]])[-12000:]
        if obj.any():
            self.fg = tt.cat([self.fg, t[tt.from_numpy(obj).cuda()]])[-4000:]

    def learn_negatives(self):
        """After a CONFIRMED sighting: strong look-alikes elsewhere in that picture teach 'not this'."""
        s, t, _, _, _ = self.last
        lab, i = self.best_label
        outside = (lab != i) & (s > 0)
        if outside.any():
            self.bg = self.torch.cat([self.bg, t[self.torch.from_numpy(outside).cuda()]])[-12000:]

    def reference(self, views):
        """The object's typical peak score when seen in a NEW picture: each taught view is scored
        against the other views only (a view matched against itself scores far too high:
        Tape 0.53 vs 0.34 held out, and the floor views only 0.13-0.15)."""
        tt = self.torch
        T = [self.patches(img) for img, _ in views]
        M = [tt.from_numpy(self.small(m > 0)).cuda() for _, m in views]
        peaks = []
        for k in range(len(views)):
            others = [j for j in range(len(views)) if j != k and M[j].any()]
            if not others or not M[k].any():
                continue
            fg = tt.cat([T[j][M[j]] for j in others])
            bg = tt.cat([T[j][~M[j]][::2] for j in others])
            sf = (T[k] @ fg.T).topk(3, dim=-1).values.mean(-1)
            sb = (T[k] @ bg.T).topk(3, dim=-1).values.mean(-1)
            peaks.append(float((sf - sb)[M[k]].max()))
        if peaks:
            self.ref_peak = float(np.median(peaks))

    def save(self, name):
        BANKS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BANKS / f"{name}.patches.npz", fg=self.fg.cpu().numpy(), bg=self.bg.cpu().numpy(),
                            ref_peak=self.ref_peak)

    def load(self, name):
        f = BANKS / f"{name}.patches.npz"
        if not f.exists():
            return False
        d = np.load(f)
        self.fg = self.torch.from_numpy(d["fg"]).cuda()
        self.bg = self.torch.from_numpy(d["bg"]).cuda()
        self.ref_peak = float(d["ref_peak"]) if "ref_peak" in d else 0.34
        self.learn_profile(taught_views(name))
        return True


def taught_views(name):
    """(picture, outline) pairs saved by `iro track teach NAME`."""
    from iro.objects import OBJ_DIR
    out = []
    for m in sorted((OBJ_DIR / name / "masks").glob("*_ann_*.png")):
        img = cv2.imread(str(OBJ_DIR / name / "images" / (m.stem + ".jpg")))
        mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if img is not None and mask is not None:
            out.append((img, mask))
    return out


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
    done, views_taught = 0, []
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
            views_taught.append((f, mask))
            save_label(d, f"{stamp}_ann_p{i:02d}", f, tight, mask)
            rec.memory.add(name, rec.fp([crop(f, tight)]))
            done += 1
            print(f"  learnt view {done}")
    except KeyboardInterrupt:
        print()
    cam.release()
    if done:
        pm.save(name)
        pm.reference(taught_views(name))
        pm.save(name)
        print(f"  its typical match strength: {pm.ref_peak:.2f} (sightings need >= {pm.ref_peak * MIN_PEAK:.2f})")
        print(f"\nLearnt {done} views of '{name}' ({len(pm.fg)} object patches). Try: iro track {name}")
    return 0 if done else 1


def gray_small(img):
    """Grey 160x120 copy for picture_shift."""
    return np.float32(cv2.cvtColor(cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))


_HANN = cv2.createHanningWindow((160, 120), cv2.CV_32F)


def picture_shift(a, b, width):
    """How far the whole picture moved from a to b (gray_small images), in pixels of a
    `width`-wide camera picture, and how sure (0..1)."""
    (dx, dy), resp = cv2.phaseCorrelate(a, b, _HANN)
    return np.array([dx, dy]) * width / 160, resp


def fresh(cam, n=3):
    """A frame certainly taken after now: skip the ones that may still sit in the camera queue."""
    for _ in range(n):
        f = cam.latest()
    return f


def flow_jacobian(arm, cam, q0, joints=(0, 3), step_deg=6):
    """How the WHOLE picture shifts per radian of each joint (2 x len(joints)), from small nudges.
    Uses the static room (phase correlation), so the object does not need to stay still.
    Each joint is nudged + and - from q0: the two shifts must point opposite ways, otherwise
    the pictures were stale or the room moved, and (None, reason) is returned."""
    J = np.zeros((2, len(joints)))
    for k, j in enumerate(joints):
        arm.move_joints(q0, duration=0.6)
        time.sleep(0.3)
        f = fresh(cam)
        base, width = gray_small(f), f.shape[1]
        d = {}
        for sign in (+1, -1):
            q = q0.copy()
            q[j] += sign * np.radians(step_deg)
            arm.move_joints(q, duration=0.6)
            time.sleep(0.3)
            d[sign], _ = picture_shift(base, gray_small(fresh(cam)), width)
        arm.move_joints(q0, duration=0.6)
        if np.dot(d[+1], d[-1]) > -0.3 * np.linalg.norm(d[+1]) * np.linalg.norm(d[-1]) or min(
                np.linalg.norm(d[+1]), np.linalg.norm(d[-1])) < 10:
            return None, f"joint {j}: +/-{step_deg} deg moved the picture {d[+1].round()} / {d[-1].round()} px (should be opposite)"
        J[:, k] = (d[+1] - d[-1]) / np.radians(2 * step_deg)
    return J, ""


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "teach":
        return cmd_teach(a[1])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
