"""ARC's body: the only things the language model may make the robot do.

The model picks a skill by name (its output is grammar-constrained, so it cannot name anything
else); this file checks the skill can run NOW (preconditions), runs it with the tested code, and
returns what happened, which ARC then tells the visitor. The model never invents a movement.
(SayCan: the model proposes, the robot checks feasibility; ROSA/ROSClaw: validate before acting.)

    iro arc routine teach NAME    pose the limp arm by hand, Enter per keyframe (greet, bow, perform)
    iro arc routine play NAME     replay it smoothly
    iro arc routine list

Skills:
    greet / bow / perform   replay a taught routine (arc/routines/NAME.yaml)
    look                    the wrist camera recognises the taught objects in view
    show:NAME               pick a taught object up, hold it up, put it back (iro pick NAME --drop)
    stop                    never reaches the model: "stop" / "berhenti" / "halt" stop everything
"""

import sys
import threading
import time

import numpy as np
import yaml

from iro import ROOT

ROUTINES = ROOT / "arc" / "routines"
STOP_WORDS = {"stop", "berhenti", "halt", "henti", "stopp", "hentikan", "jangan"}


def aka(name):
    """An object's name plus the everyday names visitors use (objects/NAME/about.yaml: names)."""
    from iro.objects import OBJ_DIR
    f = OBJ_DIR / name / "about.yaml"
    names = yaml.safe_load(f.read_text()).get("names", []) if f.exists() else []
    base = name.replace("_", " ")
    return base + (f" (also called: {', '.join(names)})" if names else "")


class Body:
    """What the robot is doing / holding / seeing: ARC gets this as a status line every turn."""

    def __init__(self, robot=False):
        self.robot = robot            # False: skills are simulated (chat / eval without the arm)
        self.busy = None
        self.seen = []                # names recognised by the last look
        self.holding = None
        self.stop_flag = threading.Event()
        self._rec = None

    # ---------------------------------------------------------------- what ARC can be told
    def state(self):
        parts = [f"arm {'busy: ' + self.busy if self.busy else 'free'}",
                 f"holding {self.holding or 'nothing'}",
                 f"last seen: {', '.join(self.seen) if self.seen else 'nothing yet'}"]
        if not self.robot:
            parts.append("body simulated")
        return "; ".join(parts)

    # ---------------------------------------------------------------- the menu
    @staticmethod
    def pickable():
        from iro.grasp import grasp_file
        from iro.objects import OBJ_DIR
        return sorted(p.name for p in OBJ_DIR.iterdir() if p.is_dir() and grasp_file(p.name).exists())

    def menu(self, actions):
        """{tag: description} the model may use: the topic's actions that exist in this file,
        show:NAME for every object with a taught grasp."""
        m = {k: v for k, v in actions.items() if k in ("greet", "bow", "perform", "look")}
        if "show" in actions:
            for n in self.pickable():
                m[f"show:{n}"] = (f"pick up the {aka(n)}, hold it up for the visitor, put it back "
                                  f"(use when asked to show or pick up this object)")
        return m

    def grammar(self, actions):
        """GBNF for llama.cpp: an optional <act>TAG</act> from the menu, then plain spoken text.
        The model cannot produce a tag that is not on the menu, and can always choose none."""
        tags = " | ".join(f'"{t}"' for t in self.menu(actions))
        return (f'root ::= act? text\nact ::= "<act>" tag "</act> "\ntag ::= {tags}\n'
                'text ::= [^<]*\n')

    # ---------------------------------------------------------------- doing
    def check(self, tag):
        """None if the skill can run now, else why not (told to the visitor honestly)."""
        name = tag.split(":")[0]
        if self.busy:
            return f"the arm is still busy ({self.busy})"
        if name in ("greet", "bow", "perform") and not (ROUTINES / f"{name}.yaml").exists():
            return f"the '{name}' movement has not been taught to the arm yet"
        if name == "show" and tag.split(":", 1)[1] not in self.pickable():
            return "that object has no taught grasp"
        return None

    def run(self, tag):
        """Run a skill; returns a short report of what happened (for ARC to tell)."""
        why = self.check(tag)
        if why:
            return f"could not do it: {why}"
        name, _, arg = tag.partition(":")
        self.busy = tag
        try:
            if not self.robot:
                time.sleep(0.5)
                return f"(simulated) did {tag}"
            if name in ("greet", "bow", "perform"):
                play(name, self.stop_flag)
                return f"did the {name} movement"
            if name == "look":
                return self.look()
            if name == "show":
                from iro.grasp import cmd_pick
                ok = cmd_pick(arg, drop=True) == 0
                return f"picked up the {arg}, showed it and put it back" if ok else \
                    f"tried to pick up the {arg} but could not (see the recording)"
            return f"unknown skill {tag}"
        except Exception as e:                    # noqa: BLE001 - tell, do not crash the conversation
            return f"could not do it: {str(e)[:120]}"
        finally:
            self.busy = None

    def look(self):
        from iro.instant import Recognizer
        from iro.objects import grab, open_camera
        if self._rec is None:
            self._rec = Recognizer()
        cam = open_camera()
        try:
            frame = grab(cam, 3)
        finally:
            cam.release()
        found = self._rec.recognize(frame)
        self.seen = [n for n, *_ in found]
        if not found:
            return "looked with the wrist camera: no object you know is in view"
        return "looked with the wrist camera and recognised: " + ", ".join(
            f"{n.replace('_', ' ')} (confidence {m:+.2f})" for n, m, *_ in found)

    def stop(self):
        self.stop_flag.set()


def is_stop(text):
    w = set(text.lower().replace("!", " ").replace(".", " ").split())
    return bool(w & STOP_WORDS) and len(w) <= 4


# ------------------------------------------------------------------ taught routines (keyframes)

def inside(arm, q, margin_deg=2.0):
    """q pulled just inside every joint's safe range (the URDF limit AND the calibrated servo range).
    Posing by hand easily reaches a mechanical stop the mover refuses (23 Oct: elbow at -98 deg,
    the limit is -96.8). Returns (q, [notes])."""
    from iro.kinematics import JOINTS
    TICKS = 4096
    q = np.array(q, float)
    notes = []
    for i, (j, (lo, hi)) in enumerate(zip(JOINTS, arm.kin.limits)):
        c = arm.cal[j]
        a, b = sorted([arm.to_rad(j, c["range_min"]), arm.to_rad(j, c["range_max"])])
        lo_, hi_ = max(lo, a) + np.radians(margin_deg), min(hi, b) - np.radians(margin_deg)
        if (b - a) >= 2 * np.pi * (TICKS - 2) / TICKS:     # no calibrated limit (wraps): URDF only
            lo_, hi_ = lo + np.radians(margin_deg), hi - np.radians(margin_deg)
        v = float(np.clip(q[i], lo_, hi_))
        if abs(v - q[i]) > 1e-6:
            notes.append(f"{j} {np.degrees(q[i]):.0f} -> {np.degrees(v):.0f} deg")
        q[i] = v
    return q, notes


def teach(name):
    """Torque off; you pose the arm; Enter saves a keyframe (type seconds first to set how long the
    move INTO this pose takes, default 1.0); q finishes. Recorded like every hardware run."""
    from iro.motion import Arm
    from iro.record import run
    if input(f"Teaching '{name}': the arm goes LIMP. Hold it. Ready? [y/N] ").strip().lower() != "y":
        return 1
    r = run(f"routine-teach-{name}")
    frames = []
    with Arm(dry_run=True) as arm:
        arm.bus.disable_torque()
        print("Pose the arm, then Enter (or a number of seconds + Enter). q + Enter to finish.")
        while True:
            a = input(f"  keyframe {len(frames) + 1}: ").strip().lower()
            if a == "q":
                break
            q, notes = inside(arm, arm.joints()[:5])
            if notes:
                print(f"    at a joint limit, saved just inside it: {', '.join(notes)}")
            secs = float(a) if a.replace(".", "", 1).isdigit() else 1.0
            frames.append({"joints_deg": np.round(np.degrees(q), 1).tolist(), "seconds": secs})
            r.log("keyframes", i=len(frames), seconds=secs, joints_deg=frames[-1]["joints_deg"])
            print(f"    saved {np.round(np.degrees(q), 0).tolist()} ({secs:.1f} s)")
    if not frames:
        return 1
    ROUTINES.mkdir(parents=True, exist_ok=True)
    (ROUTINES / f"{name}.yaml").write_text(yaml.safe_dump(
        {"name": name, "date": time.strftime("%Y-%m-%d %H:%M"), "keyframes": frames}, sort_keys=False))
    print(f"Saved arc/routines/{name}.yaml ({len(frames)} keyframes). Rest the arm. Test: iro arc routine play {name}")
    return 0


def play(name, stop_flag=None):
    """Replay keyframes with the safe mover (minimum-jerk, speed cap, limits, floor guard)."""
    from iro.motion import Arm
    from iro.record import run
    r = run(f"routine-{name}")
    k = yaml.safe_load((ROUTINES / f"{name}.yaml").read_text())["keyframes"]
    with Arm() as arm:
        for i, f in enumerate(k):
            if stop_flag is not None and stop_flag.is_set():
                r.note("stopped")
                return
            q, notes = inside(arm, np.radians(f["joints_deg"]))
            if notes:
                r.note(f"keyframe {i + 1} at a limit, moved inside: {notes}")
                print(f"  keyframe {i + 1} is at a joint limit; using just inside it: {', '.join(notes)}")
            arm.move_joints(q, duration=f["seconds"])
            r.log("played", i=i, joints_deg=np.round(np.degrees(arm.joints()), 1).tolist())


def main():
    a = sys.argv[1:]
    if a[:1] == ["teach"] and len(a) > 1:
        return teach(a[1])
    if a[:1] == ["play"] and len(a) > 1:
        play(a[1])
        return 0
    if a[:1] == ["list"]:
        for p in sorted(ROUTINES.glob("*.yaml")):
            print(f"  {p.stem}: {len(yaml.safe_load(p.read_text())['keyframes'])} keyframes")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
