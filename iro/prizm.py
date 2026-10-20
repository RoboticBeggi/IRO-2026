"""iro prizm find | backup | restore <file.hex>

The TETRIX PRIZM is Arduino-based: it stores the COMPILED program, not the
source code. `backup` copies that program (and the EEPROM) off the board, so
the old behaviour can always be put back with `restore`, and lists the text
strings inside it (serial commands, messages) that show how it worked.
"""

import glob
import os
import re
import subprocess
import sys
import time

import serial.tools.list_ports

from iro import ROOT, load_config

# (part, programmer, baud): Uno-style boards first (PRIZM is programmed as an Uno), then Mega
TRY = [("m328p", "arduino", 115200), ("m328p", "arduino", 57600), ("m2560", "wiring", 115200)]


def avrdude():
    exe = sorted(glob.glob(os.path.expanduser("~/.arduino15/packages/arduino/tools/avrdude/*/bin/avrdude")))
    if not exe:
        sys.exit("avrdude missing. Run: iro setup  (installs arduino-cli and the AVR tools)")
    conf = glob.glob(os.path.join(os.path.dirname(exe[-1]), "..", "etc", "avrdude.conf"))
    return exe[-1], conf[0] if conf else None


def known_ports(cfg):
    return {os.path.realpath(cfg["devices"][k]["port"]) for k in ("arm", "lidar") if os.path.exists(cfg["devices"][k]["port"])}


def candidates(cfg):
    taken = known_ports(cfg)
    return [p for p in serial.tools.list_ports.comports()
            if os.path.realpath(p.device) not in taken and p.device.startswith(("/dev/ttyUSB", "/dev/ttyACM"))]


def run_avrdude(port, part, prog, baud, *ops):
    exe, conf = avrdude()
    cmd = [exe, *(["-C", conf] if conf else []), "-p", part, "-c", prog, "-P", port, "-b", str(baud), "-D", *ops]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180)


def identify(port):
    """Find which bootloader settings this board answers to. Returns (part, prog, baud, signature)."""
    for part, prog, baud in TRY:
        r = run_avrdude(port, part, prog, baud)
        sig = re.search(r"[Dd]evice signature = (0x[0-9a-fA-F]+)", r.stderr + r.stdout)
        if r.returncode == 0 and sig:
            return part, prog, baud, sig.group(1)
    return None


def cmd_find(cfg):
    ports = candidates(cfg)
    if not ports:
        print("No unknown serial device. Is the PRIZM plugged into the Jetson by USB and switched on?")
        return 1
    for p in ports:
        print(f"{p.device}: {p.vid and f'{p.vid:04x}:{p.pid:04x}'} {p.manufacturer or ''} {p.product or ''} serial={p.serial_number}")
        found = identify(p.device)
        print(f"  -> {'Arduino-type board: ' + ' '.join(map(str, found)) if found else 'no AVR bootloader answered'}")
        if found:
            print(f"  Add to config/robot.yaml under devices.prizm: usb: \"{p.vid:04x}:{p.pid:04x}\"")
    return 0


def strings(data: bytes, min_len=4):
    return [m.decode() for m in re.findall(rb"[\x20-\x7e]{%d,}" % min_len, data)]


def cmd_backup(cfg):
    ports = candidates(cfg)
    if not ports:
        print("PRIZM not found. Plug it into the Jetson by USB and switch it on.")
        return 1
    port = ports[0].device
    found = identify(port)
    if not found:
        print(f"{port} did not answer as an Arduino bootloader. Press the PRIZM reset button and try again.")
        return 1
    part, prog, baud, sig = found
    out = ROOT / "backup" / "prizm" / time.strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading PRIZM on {port} ({part}, signature {sig}) into {out.relative_to(ROOT)}/ ...")
    r = run_avrdude(port, part, prog, baud, "-U", f"flash:r:{out}/flash.bin:r", "-U", f"flash:r:{out}/flash.hex:i",
                    "-U", f"eeprom:r:{out}/eeprom.hex:i")
    (out / "avrdude.log").write_text(r.stderr)
    if r.returncode != 0:
        print("Read failed, see avrdude.log:\n" + r.stderr[-600:])
        return 1
    data = (out / "flash.bin").read_bytes()
    used = len(data.rstrip(b"\xff"))
    texts = strings(data)
    (out / "strings.txt").write_text("\n".join(texts) + "\n")
    (out / "info.txt").write_text(f"port {port}\npart {part}\nprogrammer {prog} @ {baud}\nsignature {sig}\n"
                                  f"program size {used} bytes\nrestore: iro prizm restore {out.relative_to(ROOT)}/flash.hex\n")
    print(f"Saved: program {used} bytes, EEPROM, {len(texts)} text strings.")
    print("Text found inside the program (first 40):")
    for t in texts[:40]:
        print("   ", t)
    print("Run `iro backup` to send this to GitHub.")
    return 0


def cmd_restore(cfg, hexfile):
    ports = candidates(cfg)
    if not ports:
        print("PRIZM not found.")
        return 1
    found = identify(ports[0].device)
    if not found:
        print("No bootloader answer. Press reset on the PRIZM and retry.")
        return 1
    part, prog, baud, _ = found
    if input(f"Overwrite the PRIZM program with {hexfile}? Type yes: ").strip() != "yes":
        return 1
    r = run_avrdude(ports[0].device, part, prog, baud, "-U", f"flash:w:{hexfile}:i")
    print(r.stderr[-800:])
    return r.returncode


def main():
    args = sys.argv[1:] or ["find"]
    cfg = load_config()
    if args[0] == "find":
        return cmd_find(cfg)
    if args[0] == "backup":
        return cmd_backup(cfg)
    if args[0] == "restore" and len(args) == 2:
        return cmd_restore(cfg, args[1])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
