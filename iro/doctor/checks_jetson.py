"""Jetson health: power, heat, memory, disk, software."""

import shutil
import socket
import subprocess
from pathlib import Path

from iro import ROOT
from iro.hw import system

from .core import FAIL, INFO, OK, SKIP, WARN, Result, check

G = "jetson"
_last_oc: dict[str, int] = {}


@check(G, "power supply", quick=True)
def power(cfg):
    lim = cfg["limits"]["jetson"]
    rails = system.power_rails()
    if "VDD_IN" not in rails:
        return Result(G, "power supply", SKIP, "power sensor not readable")
    mv, ma = rails["VDD_IN"]
    watts = mv * ma / 1e6
    events = system.overcurrent_events()
    total = sum(events.values())
    new = total - sum(_last_oc.values()) if _last_oc else 0
    _last_oc.update(events)
    data = {"vdd_in_mv": mv, "watts": round(watts, 1), "overcurrent_events": total}
    if mv < lim["vdd_in_min_mv"]:
        return Result(G, "power supply", FAIL, f"input {mv / 1000:.2f} V is too low ({watts:.1f} W draw)",
                      "Underpowered: use the original 19 V Jetson adapter or a stronger battery regulator; "
                      "check the barrel plug is fully in.", data)
    # oc1 = input under-voltage (the SUPPLY sags); oc2 = average input power; oc3 = instantaneous
    # input power (short LOAD spikes, e.g. the GPU bursting for the LLM; too brief for the INA3221
    # to show). NVIDIA forum mapping for the Orin Nano; measured 23 Oct: oc3 only, supply steady 4.99 V.
    if events.get("oc1"):
        return Result(G, "power supply", WARN,
                      f"input voltage dipped {events['oc1']} time(s) since boot; now {mv / 1000:.2f} V, {watts:.1f} W",
                      "The SUPPLY sags under load: use the original 19 V Jetson adapter (or a stronger battery "
                      "regulator) and check the barrel plug.", data)
    if total and total < 50 and not new:            # a few spikes are normal in 25W mode (1 per minute of LLM)
        return Result(G, "power supply", INFO,
                      f"{mv / 1000:.2f} V, {watts:.1f} W; {total} brief power-spike throttle(s) since boot (normal)",
                      data=data)
    if total:
        return Result(G, "power supply", WARN,
                      f"power spikes throttled the Jetson {total} time(s) since boot"
                      + (f" ({new} just now)" if new else "") + f"; supply fine ({mv / 1000:.2f} V, {watts:.1f} W)",
                      "Not the adapter: short load spikes (GPU bursts for the LLM, vision) hit the module's "
                      "instant power limit. Compare modes with tools/power_test.sh; 25W mode avoids most of "
                      "them (sudo nvpmodel -m 1).", data)
    return Result(G, "power supply", OK, f"{mv / 1000:.2f} V, {watts:.1f} W, no power-limit events", data=data)


@check(G, "power mode")
def power_mode(cfg):
    want = cfg["limits"]["jetson"]["power_mode"]
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return Result(G, "power mode", SKIP, "nvpmodel not available")
    mode = next((line.split(":", 1)[1].strip() for line in out.splitlines() if "Power Mode" in line), "?")
    if mode != want:
        return Result(G, "power mode", WARN, f"{mode} (expected {want})",
                      f"Slower AI and voice replies. Pick {want} in the power menu (top right of the "
                      "desktop), or find its number with `nvpmodel -p --verbose` and run sudo nvpmodel -m <n>")
    return Result(G, "power mode", OK, mode)


@check(G, "temperature", quick=True)
def temperature(cfg):
    lim = cfg["limits"]["jetson"]
    temps = system.thermal_zones()
    if not temps:
        return Result(G, "temperature", SKIP, "no readable sensors")
    name, hottest = max(temps.items(), key=lambda kv: kv[1])
    fan = system.read_int(str((system.hwmon("pwm_tach") or Path("/x")) / "rpm"))
    fan_txt = f", fan {fan} rpm" if fan is not None else ""
    data = {"temps": temps, "fan_rpm": fan}
    if hottest >= lim["temp_fail_c"]:
        return Result(G, "temperature", FAIL, f"{name} at {hottest:.0f} C{fan_txt}",
                      "About to throttle hard or shut down. Stop heavy work, check the fan spins and the "
                      "heatsink is not blocked.", data)
    if hottest >= lim["temp_warn_c"]:
        return Result(G, "temperature", WARN, f"{name} at {hottest:.0f} C{fan_txt}",
                      "Running hot: give it airflow, keep it out of the sun, check the fan.", data)
    if fan == 0 and hottest > 60:
        return Result(G, "temperature", WARN, f"{hottest:.0f} C but fan not spinning",
                      "Check the fan cable on the carrier board.", data)
    return Result(G, "temperature", OK, f"hottest {name} {hottest:.0f} C{fan_txt}", data=data)


@check(G, "memory", quick=True)
def memory(cfg):
    lim = cfg["limits"]["jetson"]
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    avail = int(info["MemAvailable"].split()[0]) // 1024
    total = int(info["MemTotal"].split()[0]) // 1024
    data = {"available_mb": avail, "total_mb": total}
    msg = f"{avail} MB free of {total} MB (shared by CPU and GPU)"
    if avail < lim["ram_free_fail_mb"]:
        return Result(G, "memory", FAIL, msg, "Programs will crash or freeze. Close unused programs "
                      "(browser, VS Code) or use a smaller LLM.", data)
    if avail < lim["ram_free_warn_mb"]:
        return Result(G, "memory", WARN, msg, "Little room left for the LLM and vision models. "
                      "Close the browser and other GUI apps during runs.", data)
    return Result(G, "memory", OK, msg, data=data)


@check(G, "disk space")
def disk(cfg):
    lim = cfg["limits"]["jetson"]
    free = shutil.disk_usage("/").free / 1e9
    if free < lim["disk_free_fail_gb"]:
        return Result(G, "disk space", FAIL, f"{free:.0f} GB free", "Delete old recordings or models.")
    if free < lim["disk_free_warn_gb"]:
        return Result(G, "disk space", WARN, f"{free:.0f} GB free", "Clean up old recordings and datasets.")
    return Result(G, "disk space", OK, f"{free:.0f} GB free")


@check(G, "SSD health", slow=True)
def ssd(cfg):
    # The old SSD died once; watch for the signs. smartctl needs root, so read what sysfs offers.
    log =subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-q", "-g", "nvme.*(error|timeout|reset)"],
                         capture_output=True, text=True, timeout=10)
    errs = [line for line in log.stdout.splitlines() if line.strip()]
    if errs:
        return Result(G, "SSD health", FAIL, f"{len(errs)} NVMe error(s) in the kernel log since boot",
                      "Possible SSD failure. Run `iro backup now` immediately and plan to replace the SSD. "
                      f"Last: {errs[-1][-120:]}")
    return Result(G, "SSD health", OK, "no NVMe errors in the kernel log since boot")


@check(G, "python environment")
def venv(cfg):
    try:
        import lerobot  # noqa: F401
        return Result(G, "python environment", OK, f"LeRobot {lerobot.__version__}")
    except ImportError:
        return Result(G, "python environment", FAIL, "LeRobot not importable", "Run: iro setup")


@check(G, "GPU (CUDA)", slow=True)
def cuda(cfg):
    try:
        import torch
    except ImportError as e:
        return Result(G, "GPU (CUDA)", FAIL, f"torch not importable: {e}",
                      "Run: iro setup (env.sh must put libcudss on LD_LIBRARY_PATH)")
    if not torch.cuda.is_available():
        return Result(G, "GPU (CUDA)", FAIL, f"torch {torch.__version__} cannot see the GPU",
                      "Wrong torch build (CPU-only from PyPI?). Reinstall: iro setup")
    return Result(G, "GPU (CUDA)", OK, f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")


@check(G, "ROS 2")
def ros(cfg):
    if not Path("/opt/ros/humble/setup.bash").exists():
        return Result(G, "ROS 2", WARN, "ROS 2 Humble not installed", "Run: iro setup (needs your sudo password)")
    built = (ROOT / "ros2_ws" / "install" / "setup.bash").exists()
    if not built:
        return Result(G, "ROS 2", WARN, "ROS 2 installed, robot packages not built", "Run: iro build")
    return Result(G, "ROS 2", OK, "Humble installed, robot packages built")


@check(G, "network")
def network(cfg):
    try:
        socket.create_connection(("github.com", 443), timeout=2).close()
        return Result(G, "network", INFO, "online (GitHub reachable)")
    except OSError:
        return Result(G, "network", INFO, "offline: backups go to the USB drive only (competition mode)")
