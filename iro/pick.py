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
from pathlib import Path

import cv2
import numpy as np
import yaml

from iro import ROOT, load_config
from iro.hw.servos import BusyError
from iro.motion import Arm, MotionError

TAUGHT = ROOT / "config" / "pick_taught.yaml"   # written by `iro pick teach`, overrides robot.yaml
SNAP = Path.home() / "iro-pick-last.jpg"


def save_taught(values: dict):
    """Merge into config/pick_taught.yaml (never drop what other tools taught)."""
    data = yaml.safe_load(TAUGHT.read_text()) if TAUGHT.exists() else {}
    data.update(values)
    TAUGHT.write_text("# Written by iro pick teach / iro arm check / iro arm zero; overrides `pick:` in robot.yaml\n"
                      + yaml.safe_dump(data))


def pick_config() -> dict:
    cfg = load_config()["pick"]
    if TAUGHT.exists():
        cfg.update(yaml.safe_load(TAUGHT.read_text()) or {})
    return cfg


class Eye:
    """Wrist camera + YOLO (stock classes and your taught objects, see `iro object`)."""

    def __init__(self, cfg):
        from iro.objects import Detector, open_camera
        self.cap = open_camera()
        self.det = Detector(conf=cfg["min_conf"])
        self.cfg = cfg
        self.size = None

    def frame(self):
        from iro.objects import grab
        f = grab(self.cap)
        self.size = f.shape[1], f.shape[0]
        return f

    def classes(self):
        return list(self.cfg["classes"]) + self.det.taught_names

    def detect(self, want=None, near=None, save=True):
        """Best detection of the wanted class(es): (name, conf, (u, v) centre, box) or None.
        `near`: prefer the one closest to this pixel (tracking the same object)."""
        found, img = self.det.detect(self.frame(), conf=self.cfg["min_conf"], draw=save)
        if save:
            cv2.imwrite(str(SNAP), img)
        classes = [want] if want else self.classes()
        found = [(n, p, c, b) for n, p, c, b, _ in found if n in classes]
        if not found:
            return None
        if near is not None:
            return min(found, key=lambda d: np.linalg.norm(d[2] - near))
        return found[0]

    def everything(self, conf=0.15):
        """All detections (stock + taught), best first: [(name, conf, (u, v))]."""
        found, img = self.det.detect(self.frame(), conf=conf, draw=True)
        cv2.imwrite(str(SNAP), img)
        return [(n, p, c) for n, p, c, _, _ in found]

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
            print(f"   nothing found (classes: {args.object or eye.classes()}). What the camera saw: {SNAP}")
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
    """Look pose, then live: what does YOLO see? Use it to choose objects before teaching."""
    eye = Eye(cfg)
    with Arm(dry_run=args.dry_run) as arm:
        look_at(arm, cfg, cfg["look_xy"])
        while True:
            seen = eye.everything()
            ok = [d for d in seen if d[0] in eye.classes() and d[1] >= cfg["min_conf"]]
            print("YOLO sees: " + (", ".join(f"{n} {c:.2f}" for n, c, _ in seen[:8]) or "nothing")
                  + (f"   -> can pick: {ok[0][0]}" if ok else "   -> nothing pickable"))
            if input(f"(image: {SNAP})  Enter = look again, q = quit: ").strip().lower() == "q":
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
        expected = cfg["table_z"]   # from `iro arm zero` if it ran, else -mount_height
        print(f"   table_z = {taught['table_z']} (expected about {expected})")
        if abs(taught["table_z"] - expected) > 0.015:
            arm.hold()
            print(f"   That is {abs(taught['table_z'] - expected) * 100:.1f} cm off. Either a joint counts the wrong "
                  "way (run `iro arm check`), the joint zero errors are not corrected yet (run `iro arm zero`), "
                  "or the jaw tips were not what touched the table. Nothing saved.")
            return 1
        input("Now open the jaws by hand around an object standing on the table, at a good grasp "
              "height, gripper pointing straight down. Press Enter when it is in place. ")
        T = arm.kin.fk(arm.joints())
        tip = T[:3, 3]
        taught["grasp_z"] = round(float(tip[2]), 4)
        arm.hold()
        if taught["grasp_z"] <= taught["table_z"] or T[2, 2] > -0.9:
            print(f"   The numbers do not fit: grasp at {taught['grasp_z']} m vs table at {taught['table_z']} m, "
                  f"gripper axis {T[:3, 2].round(2)} (straight down is [0, 0, -1]).")
            print("   Either a joint counts the wrong way (run `iro arm check`), or the gripper was not pointing "
                  "straight down. Nothing saved.")
            return 1
        save_taught(taught)   # heights are good: keep them even if the camera step fails
        print(f"   grasp_z = {taught['grasp_z']} (saved). Holding. Opening and rising to the look pose above it...")
        arm.pick["table_z"] = taught["table_z"]
        arm.gripper(open=True)
        arm.move_tip([tip[0], tip[1], min(tip[2] + 0.04, cfg["lift_z"])], duration=2.0)
        look_at(arm, cfg, tip[:2])
        want = None
        while True:
            d = eye.detect(want)
            if d is not None:
                break
            seen = eye.everything()
            print(f"   YOLO does not see a pick object ({want or ', '.join(eye.classes())}). Image: {SNAP}")
            print("   It does see: " + (", ".join(f"{n} {c:.2f} at {np.round(p)}" for n, c, p in seen[:6]) or "nothing"))
            a = input("   Do NOT move the object. Enter = look again, a class name (e.g. 'book') = use that, "
                      "q = stop (heights stay saved): ").strip()
            if a.lower() == "q":
                return 1
            if a:
                want = a
                eye.cfg = dict(eye.cfg, min_conf=0.15)
        taught["grasp_pixel"] = [round(float(d[2][0]), 1), round(float(d[2][1]), 1)]
        print(f"   {d[0]} seen at pixel {taught['grasp_pixel']}: that is 'straight under the gripper'.")
    save_taught(taught)
    print(f"Saved to {TAUGHT.relative_to(ROOT)}. Next: iro pick --dry-run")
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
    except (MotionError, BusyError) as e:
        print(f"STOPPED: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C; the arm holds its pose. `iro arm off` makes it limp.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
