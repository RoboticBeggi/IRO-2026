"""iro hw snapshot: save what every component remembers into backup/hardware/<date>/.

The servos, lidar, cameras and mic array keep settings in their own memory.
If the Jetson dies, this snapshot (backed up with the repo) is how we know
exactly how they were set. Read-only on every device.
"""

import json
import os
import sys
import time

from iro import ROOT, load_config
from iro.hw import rplidar, system


def servos(cfg, out):
    from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

    from iro.hw.servos import open_bus

    d = cfg["devices"]["arm"]
    if system.users_of(d["port"]):
        return "skipped: arm port in use"
    with open_bus(d["port"], d["servo_ids"]) as bus:
        fw = bus._read_firmware_version(list(d["servo_ids"].values()))
        regs = {name: {reg: bus.read(reg, name, normalize=False) for reg in STS_SMS_SERIES_CONTROL_TABLE}
                for name in bus.motors}
    json.dump({"firmware": fw, "registers": regs}, open(out / "servos.json", "w"), indent=1)
    return f"{len(regs)} servos, all registers"


def lidar(cfg, out):
    d = cfg["devices"]["lidar"]
    if system.users_of(d["port"]):
        return "skipped: lidar port in use"
    info = rplidar.probe(d["port"], d["baud"])
    json.dump(info, open(out / "lidar.json", "w"), indent=1)
    return f"RPLidar {info['model']} serial {info['serial']}"


def usb_devices(cfg, out):
    import serial.tools.list_ports
    devs = []
    for dev in sorted(os.listdir("/sys/bus/usb/devices")):
        p = f"/sys/bus/usb/devices/{dev}"
        if not os.path.exists(f"{p}/idVendor"):
            continue
        info = {"path": dev}
        for f in ("idVendor", "idProduct", "bcdDevice", "manufacturer", "product", "serial", "speed"):
            try:
                info[f] = open(f"{p}/{f}").read().strip()
            except OSError:
                pass
        devs.append(info)
    ports = [{"device": p.device, "vid_pid": f"{p.vid:04x}:{p.pid:04x}" if p.vid else None,
              "serial": p.serial_number, "product": p.product} for p in serial.tools.list_ports.comports()]
    json.dump({"usb": devs, "serial_ports": ports}, open(out / "usb.json", "w"), indent=1)
    return f"{len(devs)} USB devices (firmware versions in bcdDevice)"


def oak(cfg, out):
    import depthai as dai
    found = dai.Device.getAllAvailableDevices()
    if not found:
        return "skipped: OAK-D not accessible (udev rule missing? run iro setup)"
    with dai.Device() as dev:
        calib = dev.readCalibration()
        json.dump(calib.eepromToJson(), open(out / "oak_calibration.json", "w"), indent=1)
        name = dev.getDeviceName() if hasattr(dev, "getDeviceName") else "OAK"
    return f"{name} factory calibration"


def main():
    cfg = load_config()
    out = ROOT / "backup" / "hardware" / time.strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving hardware snapshot to {out.relative_to(ROOT)}/")
    for name, fn in (("servos", servos), ("lidar", lidar), ("usb", usb_devices), ("oak-d", oak)):
        try:
            print(f"  {name:<7} {fn(cfg, out)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<7} FAILED: {type(e).__name__}: {e}")
    print("Done. It goes to GitHub/USB with the next backup (iro backup).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
