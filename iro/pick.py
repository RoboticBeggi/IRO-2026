"""iro pick: the arm finds an object with the wrist camera + YOLO and picks it up.

    iro pick                 pick the most confident object of the configured classes
    iro pick bottle          pick a bottle
    iro pick --dry-run       detect and plan, print the moves, do not move
    iro pick --drop          put it back down where it was after lifting
    iro pick look            go to the look pose and show what YOLO sees (no picking)
    iro pick teach           one-time setup with your hands on the arm (table height,
                             grasp height, and where "under the gripper" is in the image)

How a pick works (no camera calibration needed):
  1. look pose: camera straight down over the work area, YOLO finds the object
  2. two small test moves tell how the image shifts when the arm moves (image Jacobian)
  3. visual servoing: move sideways until the object sits at the taught grasp pixel,
     i.e. straight under the gripper
  4. open, descend, close, check something is in the jaws, lift
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from iro import ROOT, load_config
from iro.motion import Arm, MotionError

TAUGHT = ROOT / "config" / "pick_taught.yaml"   # written by `iro pick teach`, overrides robot.yaml
SNAP = Path.home() / "iro-pick-last.jpg"


def pick_config() -> dict:
    cfg = load_config()["pick"]
    if TAUGHT.exists():
        cfg.update(yaml.safe_load(TAUGHT.read_text()) or {})
    return cfg


class Eye:
    """Wrist camera + YOLO."""

    def __init__(self, cfg):
        from ultralytics import YOLO
        dev = load_config()["devices"]["arm_camera"]["device"]
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.model = YOLO(str(ROOT / "models" / cfg["model"]))
        self.cfg = cfg
        self.size = None

    def frame(self):
        for _ in range(4):           # drop frames buffered while the arm was moving
            self.cap.grab()
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("wrist camera gives no image")
        self.size = f.shape[1], f.shape[0]
        return f

    def detect(self, want=None, near=None, save=True):
        """Best detection of the wanted class(es): (name, conf, (u, v) centre, box) or None.
        `near`: prefer the one closest to this pixel (tracking the same object)."""
        f = self.frame()
        r = self.model.predict(f, device=0, verbose=False, conf=self.cfg["min_conf"])[0]
        classes = [want] if want else self.cfg["classes"]
        found = []
        for box, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls, r.boxes.conf):
            name = r.names[int(c)]
            if name in classes:
                u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                found.append((name, float(p), np.array([u, v]), box))
        if save:
            cv2.imwrite(str(SNAP), r.plot())
        if not found:
            return None
        if near is not None:
            return min(found, key=lambda d: np.linalg.norm(d[2] - near))
        return max(found, key=lambda d: d[1])

    def grasp_pixel(self):
        gp = self.cfg.get("grasp_pixel")
        return np.array(gp, float) if gp else np.array(self.size, float) / 2


def look_at(arm, cfg, xy):
    """Look pose with the tip above xy (lean outward so the wrist can reach the height)."""
    return arm.move_tip([xy[0], xy[1], cfg["look_z"]], tilt=np.radians(cfg["look_tilt_deg"]))


def estimate_jacobian(arm, eye, cfg, want, xy0, p0, step=0.02):
    """Pixels moved per metre of tip motion in x and y (2x2), from two small test moves."""
    J = np.zeros((2, 2))
    for i in range(2):
        xy = np.array(xy0, float)
        xy[i] += step
        look_at(arm, cfg, xy)
        d = eye.detect(want, near=p0)
        if d is None:
            raise MotionError("lost the object during the test move; is it near the edge of the image?")
        J[:, i] = (d[2] - p0) / step
        look_at(arm, cfg, xy0)
    if abs(np.linalg.det(J)) < 1e3:
        raise MotionError(f"image barely moves when the arm moves (J={np.round(J)}); check the camera")
    return J


def do_pick(args, cfg):
    eye = Eye(cfg)
    with Arm(dry_run=args.dry_run) as arm:
        look = np.array(cfg["look_xy"], float)
        print(f"1. look pose above {look}, {cfg['look_z']} m up")
        arm.gripper(open=True)
        look_at(arm, cfg, look)
        d = eye.detect(args.object)
        if d is None:
            print(f"   nothing found (classes: {args.object or cfg['classes']}). What the camera saw: {SNAP}")
            return 1
        name, conf, p, _ = d
        print(f"   found {name} ({conf:.2f}) at pixel {np.round(p)}")
        if args.dry_run:
            print(f"   [dry run] would centre it on grasp pixel {np.round(eye.grasp_pixel())}, "
                  f"descend to z={cfg['grasp_z']}, close, lift to z={cfg['lift_z']}")
            return 0

        print("2. measuring how the image moves with the arm")
        J = estimate_jacobian(arm, eye, cfg, name, look, p)
        Jinv = np.linalg.inv(J)

        print("3. centring the object under the gripper")
        xy, target = look.copy(), eye.grasp_pixel()
        for it in range(15):
            d = eye.detect(name, near=p)
            if d is None:
                raise MotionError("lost the object while centring")
            p = d[2]
            err = p - target
            print(f"   step {it}: pixel error {np.round(err)}")
            if np.linalg.norm(err) < cfg["servo_tolerance_px"]:
                break
            move = -0.7 * Jinv @ err                      # 0.7: approach without overshoot
            xy += np.clip(move, -0.04, 0.04)
            look_at(arm, cfg, xy)
        else:
            raise MotionError("could not centre the object in 15 steps")

        print("4. grasp")
        x, y = xy
        arm.move_tip([x, y, min(cfg["grasp_z"] + 0.04, cfg["lift_z"])])
        arm.move_tip([x, y, cfg["grasp_z"]], duration=1.5)
        arm.gripper(open=False, wait=1.0)
        held = arm.gripper_pct()
        arm.move_tip([x, y, cfg["lift_z"]], duration=1.5)
        if held < cfg["gripper_closed_pct"] + 3:
            print(f"   missed: the gripper closed all the way ({held:.0f}%). Last image: {SNAP}")
            look_at(arm, cfg, look)
            return 1
        print(f"   holding the {name} (gripper at {held:.0f}%)")
        if args.drop:
            arm.move_tip([x, y, cfg["grasp_z"] + 0.005], duration=1.5)
            arm.gripper(open=True)
            arm.move_tip([x, y, cfg["lift_z"]])
            print("   put it back down")
        return 0


def do_look(args, cfg):
    eye = Eye(cfg)
    with Arm(dry_run=args.dry_run) as arm:
        look_at(arm, cfg, cfg["look_xy"])
        d = eye.detect(args.object)
    print(f"Saw: {d[0]} ({d[1]:.2f}) at pixel {np.round(d[2])}" if d else "Saw none of the pick classes.")
    print(f"Image with all detections: {SNAP}")
    return 0


def do_teach(args, cfg):
    eye = Eye(cfg)
    taught = {}
    with Arm() as arm:
        print("TEACH. The arm will go limp: hold it.")
        input("Press Enter, then lower the gripper tip until it just touches the table... ")
        arm.limp()
        input("Tip touching the table? Press Enter. ")
        taught["table_z"] = round(float(arm.tip()[2]), 4)
        print(f"   table_z = {taught['table_z']}")
        input("Now open the jaws by hand around an object standing on the table, at a good grasp "
              "height, gripper pointing straight down. Press Enter when it is in place. ")
        tip = arm.tip()
        taught["grasp_z"] = round(float(tip[2]), 4)
        arm.hold()
        print(f"   grasp_z = {taught['grasp_z']}. Holding. Opening and rising to the look pose above it...")
        arm.pick["table_z"] = taught["table_z"]
        arm.gripper(open=True)
        arm.move_tip([tip[0], tip[1], min(tip[2] + 0.04, cfg["lift_z"])], duration=2.0)
        look_at(arm, cfg, tip[:2])
        d = eye.detect()
        if d is None:
            print(f"   YOLO cannot see the object from above (image: {SNAP}). Try a COCO object like a bottle or cup.")
            return 1
        taught["grasp_pixel"] = [round(float(d[2][0]), 1), round(float(d[2][1]), 1)]
        print(f"   {d[0]} seen at pixel {taught['grasp_pixel']}: that is 'straight under the gripper'.")
    TAUGHT.write_text("# Written by `iro pick teach`; overrides `pick:` in robot.yaml\n" + yaml.safe_dump(taught))
    print(f"Saved to {TAUGHT.relative_to(ROOT)}. Back it up: iro backup")
    return 0


def main():
    p = argparse.ArgumentParser(prog="iro pick", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("object", nargs="?", help="YOLO class to pick, or: look | teach")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--drop", action="store_true")
    a = p.parse_args()
    cfg = pick_config()
    mode = a.object if a.object in ("look", "teach") else "pick"
    if mode != "pick":
        a.object = None
    try:
        return {"pick": do_pick, "look": do_look, "teach": do_teach}[mode](a, cfg)
    except MotionError as e:
        print(f"STOPPED: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C; the arm holds its pose. `iro arm off` makes it limp.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
