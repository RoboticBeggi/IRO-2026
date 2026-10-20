"""Read-only health check for the SO-ARM101 follower.

Pings the servo bus, checks that IDs 1-6 are present and are STS3215s, and
prints raw position, temperature and voltage for each joint. Never changes
torque or writes to the servos, so it is safe to run at any time.

    source env.sh
    python arm/scan_motors.py [--port /dev/so101]
"""

import argparse
import os
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Same names and IDs as LeRobot's so101_follower
JOINTS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
STS3215_MODEL = 777
HOT_C = 55  # STS3215 limits itself at 70 C; warn well before


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default=os.environ.get("SO101_PORT", "/dev/ttyACM0"))
    args = parser.parse_args()

    from iro.hw.system import users_of
    if users_of(args.port):
        pid, name = users_of(args.port)[0]
        print(f"The arm is in use by {name} (pid {pid}); its health is on /diagnostics. `iro stop` frees it.")
        return 1
    bus = FeetechMotorsBus(
        port=args.port,
        motors={name: Motor(id_, "sts3215", MotorNormMode.RANGE_M100_100) for name, id_ in JOINTS.items()},
    )
    try:
        bus.connect(handshake=False)
    except Exception as e:  # noqa: BLE001
        print(f"Cannot open {args.port}: {e}")
        print("Check the USB cable, the arm's power supply, and that you are in the dialout group.")
        return 1

    try:
        found = bus.broadcast_ping(num_retry=2) or {}
        print(f"Port {args.port}: {len(found)} servo(s) answered: {sorted(found)}")

        ok = True
        for name, id_ in JOINTS.items():
            if id_ not in found:
                print(f"  MISSING  id {id_} ({name})")
                ok = False
            elif found[id_] != STS3215_MODEL:
                print(f"  WRONG MODEL id {id_} ({name}): model {found[id_]}, expected STS3215 ({STS3215_MODEL})")
                ok = False
        extra = sorted(set(found) - set(JOINTS.values()))
        if extra:
            print(f"  UNEXPECTED ids {extra}: two servos may share an ID, or one was never set")
            ok = False
        if not found:
            print("NO servo answers: the arm's 12 V power supply is off or unplugged, or the cable from "
                  "the board to the first (base) servo is loose. The USB side is fine.")
            return 1
        if not ok:
            print("Some answer, some don't: check the cable into the first missing servo (the chain runs "
                  "base -> gripper). Wrong IDs only if a servo was replaced: lerobot-setup-motors.")
            return 1

        pos = bus.sync_read("Present_Position", normalize=False)
        temp = bus.sync_read("Present_Temperature", normalize=False)
        volt = bus.sync_read("Present_Voltage", normalize=False)
        print(f"\n  {'joint':<14}{'id':>3}{'raw pos':>9}{'temp C':>8}{'volts':>7}")
        for name, id_ in JOINTS.items():
            flag = "  HOT" if temp[name] >= HOT_C else ""
            print(f"  {name:<14}{id_:>3}{pos[name]:>9}{temp[name]:>8}{volt[name] / 10:>7.1f}{flag}")
        print("\nAll 6 servos OK. Raw positions run 0-4095; about 2048 is mid-range.")
        return 0
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    sys.exit(main())
