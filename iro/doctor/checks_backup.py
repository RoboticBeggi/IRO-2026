"""Is the work safe? Snapshot age, GitHub push, USB copy, cron."""

import json
import subprocess
import time
from pathlib import Path

from iro import ROOT

from .core import FAIL, INFO, OK, WARN, Result, check

G = "backup"
STATE = Path.home() / ".iro-backup-state.json"  # written by tools/backup.sh


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10).stdout.strip()


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


@check(G, "automatic backup")
def cron(cfg):
    tab = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    if "tools/backup.sh" not in tab:
        return Result(G, "automatic backup", FAIL, "not scheduled", "Run: iro backup install")
    return Result(G, "automatic backup", OK, "scheduled (cron)")


@check(G, "latest snapshot")
def snapshot(cfg):
    branch = f"autosave/{_git('config', '--get', 'iro.host') or subprocess.getoutput('hostname')}"
    ts = _git("log", "-1", "--format=%ct", branch)
    if not ts:
        return Result(G, "latest snapshot", FAIL, "no snapshot yet", "Run: iro backup")
    age_min = (time.time() - int(ts)) / 60
    dirty = _git("status", "--porcelain")
    limit = cfg["limits"]["backup"]["snapshot_max_age_min"]
    if dirty and age_min > limit:
        return Result(G, "latest snapshot", WARN, f"{age_min:.0f} min old, with unsaved changes since",
                      "Run: iro backup   (cron may have stopped; check `iro backup log`)")
    return Result(G, "latest snapshot", OK, f"{age_min:.0f} min old")


@check(G, "GitHub copy")
def github(cfg):
    st = _state()
    last = st.get("last_push_ok")
    if not last:
        return Result(G, "GitHub copy", WARN, "never pushed successfully", "Run: iro backup  (needs internet and the deploy key)")
    age_h = (time.time() - last) / 3600
    if age_h > cfg["limits"]["backup"]["push_max_age_h"]:
        return Result(G, "GitHub copy", WARN, f"last push {age_h:.0f} h ago",
                      "Offline for a while: make sure the USB copy is current (iro backup usb).")
    return Result(G, "GitHub copy", OK, f"last push {age_h * 60:.0f} min ago")


@check(G, "USB copy")
def usb(cfg):
    st = _state()
    last = st.get("last_usb_ok")
    label = cfg["backup"]["usb_label"]
    mounted = subprocess.run(["findmnt", "-n", "-S", f"LABEL={label}"], capture_output=True).returncode == 0
    when = f"last copy {(time.time() - last) / 3600:.0f} h ago" if last else "never copied"
    if mounted:
        return Result(G, "USB copy", OK, f"{label} drive plugged in; {when}")
    return Result(G, "USB copy", INFO if last else WARN, f"{label} drive not plugged in; {when}",
                  "" if last else f"Format a USB drive as ext4 or exFAT named {label} and plug it in. "
                  "It is the only backup that works without internet (competition).")
