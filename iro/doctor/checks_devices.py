"""Is every sensor and controller plugged in, allowed, and answering?"""

import os
import subprocess

from iro.hw import rplidar, system

from .core import FAIL, INFO, OK, WARN, Result, check

G = "devices"


def _busy(port: str) -> Result | None:
    users = [u for u in system.users_of(port) if u[0] != os.getpid()]
    if users:
        names = ", ".join(f"{cmd} (pid {pid})" for pid, cmd in users)
        return Result(G, port, INFO, f"in use by {names}; not probed directly")
    return None


def _usb(name: str, ids, fix: str) -> Result | None:
    ids = [ids] if isinstance(ids, str) else ids
    if not system.usb_ids() & set(ids):
        return Result(G, name, FAIL, "not detected on USB", fix)
    return None


@check(G, "arm servo board")
def arm_board(cfg):
    d = cfg["devices"]["arm"]
    if r := _usb("arm servo board", d["usb"], "Check the arm's USB cable to the Jetson (the arm needs its own power too)."):
        return r
    if not os.path.exists(d["port"]):
        return Result(G, "arm servo board", WARN, f"on USB but {d['port']} missing",
                      "udev rule not installed: run iro setup")
    if not os.access(d["port"], os.R_OK | os.W_OK):
        return Result(G, "arm servo board", FAIL, f"no permission for {d['port']}",
                      "Run iro setup, then log out and back in (dialout group).")
    return Result(G, "arm servo board", OK, f"{d['port']}")


@check(G, "lidar")
def lidar(cfg):
    d = cfg["devices"]["lidar"]
    if r := _usb("lidar", d["usb"], "Check the lidar's USB cable and its adapter board."):
        return r
    if r := _busy(d["port"]):
        r.name = "lidar"
        return r
    try:
        info = rplidar.probe(d["port"], d["baud"])
    except Exception as e:  # noqa: BLE001
        return Result(G, "lidar", FAIL, f"no answer on {d['port']} at {d['baud']} baud: {e}",
                      "Unplug and replug the lidar; check its adapter's power LED. Wrong model? check baud in config/robot.yaml.")
    desc = f"RPLidar {info['model']} fw {info['firmware']}, health {info['health']}"
    if info["health"] != "good":
        # Same as Slamtec's driver: a latched error (e.g. after a protection stop) clears with a reset
        code = info["error_code"]
        rplidar.reset(d["port"], d["baud"])
        info = rplidar.probe(d["port"], d["baud"])
        if info["health"] == "good":
            return Result(G, "lidar", WARN, f"RPLidar {info['model']} reported error code {code}; cleared by a reset",
                          "Usually a protection stop (motor blocked, or power dipped at start). "
                          "If it keeps happening: check the lidar's power and that nothing touches the spinning head.", info)
        desc = f"RPLidar {info['model']} fw {info['firmware']}, health {info['health']} even after reset"
        return Result(G, "lidar", FAIL if info["health"] == "error" else WARN,
                      f"{desc} (code {info['error_code']})",
                      "Power-cycle the lidar. If it repeats: check the motor spins freely and the lens is clean.", info)
    return Result(G, "lidar", OK, desc, data=info)


@check(G, "arm camera")
def arm_camera(cfg):
    d = cfg["devices"]["arm_camera"]
    if r := _usb("arm camera", d["usb"], "Check the wrist camera's USB cable (it runs along the arm; look for strain at the joints)."):
        return r
    if r := _busy(d["device"]):
        r.name = "arm camera"
        return r
    import cv2
    cap = cv2.VideoCapture(d["device"], cv2.CAP_V4L2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return Result(G, "arm camera", FAIL, f"{d['device']} gives no image", "Replug the camera; check config/robot.yaml device path.")
    brightness = float(frame.mean())
    detail = f"{frame.shape[1]}x{frame.shape[0]}, brightness {brightness:.0f}/255"
    if brightness < 8:
        return Result(G, "arm camera", WARN, detail + " (image almost black)", "Lens cap on, or pointing into something dark?")
    return Result(G, "arm camera", OK, detail)


@check(G, "depth camera")
def depth_camera(cfg):
    ids = cfg["devices"]["depth_camera"]["usb"]
    if r := _usb("depth camera", ids, "Check the OAK-D USB-C cable; use a USB 3 port (blue)."):
        return r
    for dev in os.listdir("/sys/bus/usb/devices"):
        p = f"/sys/bus/usb/devices/{dev}"
        try:
            vid = open(f"{p}/idVendor").read().strip()
        except OSError:
            continue
        if vid == "03e7":
            bus, num = int(open(f"{p}/busnum").read()), int(open(f"{p}/devnum").read())
            node = f"/dev/bus/usb/{bus:03d}/{num:03d}"
            if not os.access(node, os.W_OK):
                return Result(G, "depth camera", WARN, "OAK-D detected but not usable without root",
                              "udev rule for the OAK-D missing: run iro setup")
            speed = open(f"{p}/speed").read().strip()
            if speed in ("480", "12"):
                return Result(G, "depth camera", WARN, f"OAK-D on a USB 2 link ({speed} Mbit/s)",
                              "Use a USB 3 port/cable, or depth will lag and drop frames.")
            return Result(G, "depth camera", OK, f"OAK-D detected ({speed} Mbit/s)")
    return Result(G, "depth camera", OK, "OAK-D detected")


@check(G, "mic array")
def mic(cfg):
    d = cfg["devices"]["mic_array"]
    if r := _usb("mic array", d["usb"], "Check the reSpeaker's USB cable."):
        return r
    cards = open("/proc/asound/cards").read()
    if d["alsa_name"] not in cards:
        return Result(G, "mic array", FAIL, "on USB but no sound card", "Replug it; check `arecord -l`.")
    card = next(line.split()[0] for line in cards.splitlines() if d["alsa_name"] in line)
    rec = subprocess.run(["arecord", "-D", f"plughw:{card}", "-d", "1", "-f", "S16_LE", "-r", "16000", "-q", "/dev/null"],
                         capture_output=True, text=True, timeout=10)
    if rec.returncode != 0:
        if "busy" in rec.stderr.lower():
            return Result(G, "mic array", INFO, f"sound card {card} busy (in use by the voice program)")
        return Result(G, "mic array", FAIL, f"sound card {card} will not record: {rec.stderr.strip()[:100]}",
                      "Replug; if it persists, the XVF3800 firmware may need reflashing.")
    return Result(G, "mic array", OK, f"reSpeaker XVF3800 records (card {card})")


@check(G, "PRIZM controller")
def prizm(cfg):
    d = cfg["devices"]["prizm"]
    if d.get("usb") and d["usb"] in system.usb_ids():
        return Result(G, "PRIZM controller", OK, f"connected ({d['port']})")
    unknown = [p.device for p in system.unknown_serial_ports(cfg)]
    if unknown:
        return Result(G, "PRIZM controller", WARN, f"unidentified serial port(s): {', '.join(unknown)}",
                      "Possibly the PRIZM. Run `iro prizm find` to identify it.")
    return Result(G, "PRIZM controller", INFO, "not connected")


@check(G, "base controller")
def base_controller(cfg):
    d = cfg["devices"]["base_controller"]
    if d["usb"] not in system.usb_ids():
        return Result(G, "base controller", WARN, "ESP32-S3 not on USB",
                      "Plug the base controller's USB-C into the Jetson.")
    return Result(G, "base controller", INFO,
                  "ESP32-S3 on USB (old PUSAKA motor firmware; its command link is not over USB, see backup/esp32s3/)")
