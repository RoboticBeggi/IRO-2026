"""Linux / Jetson helpers: USB devices, who holds a port, sysfs sensors."""

import os
from pathlib import Path


def usb_ids() -> set[str]:
    """All connected USB devices as 'vvvv:pppp'."""
    ids = set()
    for dev in Path("/sys/bus/usb/devices").iterdir():
        try:
            vid = (dev / "idVendor").read_text().strip()
            pid = (dev / "idProduct").read_text().strip()
        except OSError:
            continue
        ids.add(f"{vid}:{pid}")
    return ids


def users_of(path: str) -> list[tuple[int, str]]:
    """Processes that have `path` open, as (pid, command name)."""
    target = os.path.realpath(path)
    users = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                    users.append((int(pid), Path(f"/proc/{pid}/comm").read_text().strip()))
                    break
        except OSError:
            continue  # process ended, or not ours
    return users


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def hwmon(name: str) -> Path | None:
    """sysfs folder of the hwmon device called `name` (ina3221, soctherm_oc, pwmfan...)."""
    for h in Path("/sys/class/hwmon").iterdir():
        try:
            if (h / "name").read_text().strip() == name:
                return h
        except OSError:
            continue
    return None


def thermal_zones() -> dict[str, float]:
    """Readable thermal zones in degrees C (some Orin zones refuse reads; skipped)."""
    temps = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            name = (z / "type").read_text().strip()
            temps[name] = int((z / "temp").read_text()) / 1000
        except (OSError, ValueError, TypeError):  # Orin cv*-thermal zones fail reads with EAGAIN
            continue
    return temps


def power_rails() -> dict[str, tuple[int, int]]:
    """Jetson INA3221 rails: {label: (millivolts, milliamps)}."""
    h = hwmon("ina3221")
    rails = {}
    if not h:
        return rails
    for label in sorted(h.glob("in*_label")):
        n = label.name[2:-6]  # in1_label -> 1
        mv, ma = read_int(h / f"in{n}_input"), read_int(h / f"curr{n}_input")
        if mv is not None and ma is not None:
            rails[label.read_text().strip()] = (mv, ma)
    return rails


def overcurrent_events() -> dict[str, int]:
    """How many times the Jetson throttled itself for drawing too much power, per limit."""
    h = hwmon("soctherm_oc")
    if not h:
        return {}
    return {f.name.split("_")[0]: read_int(f) or 0 for f in sorted(h.glob("oc*_event_cnt"))}
