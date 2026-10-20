"""SO-ARM101: servos present, powered, cool, error-free, and correctly calibrated."""

import json
import os

from iro import ROOT
from iro.hw import system

from .core import FAIL, INFO, OK, SKIP, WARN, Result, check

G = "arm"
CAL_FILE = ROOT / "arm" / "calibration" / "robots" / "so_follower" / f"{os.environ.get('SO101_ID', 'iro_follower')}.json"


def _bus_or_skip(cfg):
    d = cfg["devices"]["arm"]
    if not os.path.exists(d["port"]):
        return None, Result(G, "servos", SKIP, f"{d['port']} not present (see devices)")
    users = [u for u in system.users_of(d["port"]) if u[0] != os.getpid()]
    if users:
        return None, Result(G, "servos", INFO, f"arm port in use by {users[0][1]} (pid {users[0][0]}); "
                            "check its health on the /diagnostics topic instead")
    from iro.hw.servos import open_bus
    return open_bus(d["port"], d["servo_ids"]), None


@check(G, "servos", quick=True)
def servos(cfg):
    from iro.hw.servos import STS3215_MODEL, read_health

    bus_cm, skip = _bus_or_skip(cfg)
    if skip:
        return skip
    lim = cfg["limits"]["servos"]
    ids = cfg["devices"]["arm"]["servo_ids"]
    results = []
    with bus_cm as bus:
        found = bus.broadcast_ping(num_retry=2) or {}
        missing = [n for n, i in ids.items() if i not in found]
        extra = sorted(set(found) - set(ids.values()))
        if missing:
            # The bus is a daisy chain from the board outward: the first missing servo is where to look.
            first = min(missing, key=ids.get)
            return Result(G, "servos", FAIL, f"not answering: {', '.join(f'{n} (id {ids[n]})' for n in missing)}",
                          f"Arm power on? If several are missing, the cable INTO {first} (id {ids[first]}) is the "
                          "likely break. One alone: its cable or the servo.")
        if extra:
            return Result(G, "servos", FAIL, f"unexpected ids {extra}",
                          "Two servos share an ID or a new servo is unset. Connect one at a time: lerobot-setup-motors")
        wrong = [n for n in ids if found[ids[n]] != STS3215_MODEL]
        if wrong:
            results.append(Result(G, "servo models", WARN, f"not STS3215: {wrong}", "Check the replacement servo type."))
        health = read_health(bus)

    volts = [h["volts"] for h in health.values()]
    vmin = min(volts)
    data = {"health": health}
    if vmin < lim["volt_fail"]:
        results.append(Result(G, "servo power", FAIL, f"{vmin:.1f} V (lowest servo)",
                              "Underpowered arm: weak or wrong supply, or loose barrel plug. The arm will droop and "
                              "jerk. Use the 12 V supply that came with the arm.", data))
    elif vmin < lim["volt_warn"]:
        results.append(Result(G, "servo power", WARN, f"{vmin:.1f} V (lowest servo)",
                              "Supply sagging. Check the plug and the supply rating (12 V, 5 A or more).", data))
    elif max(volts) > lim["volt_max"]:
        results.append(Result(G, "servo power", FAIL, f"{max(volts):.1f} V, above the servo limit",
                              "Wrong power supply: unplug it now.", data))
    else:
        results.append(Result(G, "servo power", OK, f"{vmin:.1f}-{max(volts):.1f} V", data=data))

    for name, h in health.items():
        if h["errors"]:
            results.append(Result(G, name, FAIL, f"servo reports: {', '.join(h['errors'])}",
                                  "Torque off and let it rest. Overload/over-current: something blocks the joint "
                                  "or the pose is too heavy (puppet too far out). Overheated: let it cool."))
        elif h["temp_c"] >= lim["temp_fail_c"]:
            results.append(Result(G, name, FAIL, f"{h['temp_c']} C",
                                  "About to shut itself off at 70 C. Go to rest pose with torque off now."))
        elif h["temp_c"] >= lim["temp_warn_c"]:
            results.append(Result(G, name, WARN, f"{h['temp_c']} C",
                                  "Getting hot. Rest with torque off between performances."))
        elif h["load_pct"] >= lim["load_warn_pct"]:
            results.append(Result(G, name, WARN, f"load {h['load_pct']:.0f}%",
                                  "Straining: the pose or what it holds is too heavy, or it is pushing against something."))
    hottest = max(health.items(), key=lambda kv: kv[1]["temp_c"])
    torque = sum(h["torque"] for h in health.values())
    results.append(Result(G, "servos", OK, f"all 6 answer; hottest {hottest[0]} {hottest[1]['temp_c']} C; "
                          f"torque on {torque}/6"))
    return results


@check(G, "calibration")
def calibration(cfg):
    if not CAL_FILE.exists():
        return Result(G, "calibration", FAIL, f"no calibration file ({CAL_FILE.relative_to(ROOT)})",
                      "Run: iro arm calibrate")
    cal = json.loads(CAL_FILE.read_text())
    results = []
    continuous = set(cfg["devices"]["arm"].get("continuous_joints", []))
    no_limits = [j for j, c in cal.items() if j not in continuous and c["range_min"] <= 0 and c["range_max"] >= 4095]
    if no_limits:
        results.append(Result(G, "calibration", FAIL,
                              f"no joint limits recorded for {', '.join(no_limits)} (range 0-4095)",
                              "These joints were not moved end to end during calibration, or their reading wrapped "
                              "past 4095 because the 'middle' pose was off. Recalibrate: iro arm calibrate, with the "
                              "arm in the middle pose shown in arm/README.md, then move EVERY joint end to end."))

    bus_cm, skip = _bus_or_skip(cfg)
    if skip:
        results.append(Result(G, "calibration matches servos", SKIP, skip.detail))
        return results
    with bus_cm as bus:
        on_servo = {
            n: (bus.read("Homing_Offset", n, normalize=False), bus.read("Min_Position_Limit", n, normalize=False),
                bus.read("Max_Position_Limit", n, normalize=False))
            for n in bus.motors
        }
    mismatch = [n for n, (off, lo, hi) in on_servo.items()
                if n in cal and (off, lo, hi) != (cal[n]["homing_offset"], cal[n]["range_min"], cal[n]["range_max"])]
    if mismatch:
        results.append(Result(G, "calibration matches servos", FAIL, f"file and servos disagree: {mismatch}",
                              "The servos were recalibrated elsewhere, or a servo was swapped. Run iro arm calibrate "
                              "(press Enter to write the file's values back, or c to recalibrate)."))
    else:
        results.append(Result(G, "calibration matches servos", OK, f"{CAL_FILE.name} matches all servos"))
    return results
