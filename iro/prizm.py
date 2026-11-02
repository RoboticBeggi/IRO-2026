"""iro prizm ping | backup | restore <flash_full.bin>

The TETRIX PRIZM Pro is an ESP32-S3. It stores the COMPILED program, not the
source. `backup` copies the whole 8 MB flash (program, settings, files) into
backup/esp32s3/<date>/ so the board can be put back exactly with `restore`.
`ping` checks the motor program answers (press the green START button first:
PRIZM Pro programs wait for it after every reset).
"""

import re
import subprocess
import sys
import time

from iro import ROOT, load_config
from iro.hw.system import usb_ids


def port(cfg):
    import serial.tools.list_ports
    vid, pid = (int(x, 16) for x in cfg["devices"]["base_controller"]["usb"].split(":"))
    for p in serial.tools.list_ports.comports():
        if (p.vid, p.pid) == (vid, pid):
            return p.device
    sys.exit("PRIZM Pro not found on USB. Plug in its USB-C and switch it on.")


def esptool(*args):
    return subprocess.run([sys.executable, "-m", "esptool", *args], capture_output=True, text=True, timeout=600)


def cmd_ping(cfg):
    import serial
    # Keep pyserial's default DTR=RTS=on: setting them off passes through RTS-on/DTR-off,
    # which the ESP32-S3 USB port treats as a reset (and the program then waits for START again)
    s = serial.Serial(port(cfg), 115200, timeout=0.5)
    s.reset_input_buffer()
    s.write(b"PING\n")
    reply = s.read(64).decode(errors="replace").strip()
    s.close()
    if "PONG" in reply:
        print("PONG: motor program running and listening")
        return 0
    print(f"No PONG (got {reply!r}). Press the green START button on the PRIZM Pro, then try again.")
    return 1


def cmd_backup(cfg):
    p = port(cfg)
    out = ROOT / "backup" / "esp32s3" / time.strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading all 8 MB from the PRIZM Pro on {p} (about 40 s)...")
    r = esptool("--port", p, "-b", "921600", "read-flash", "0", "ALL", str(out / "flash_full.bin"))
    (out / "esptool.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        print("Failed, see esptool.log:\n" + (r.stdout + r.stderr)[-600:])
        return 1
    data = (out / "flash_full.bin").read_bytes()
    texts = [m.decode() for m in re.findall(rb"[\x20-\x7e]{5,}", data[0x10000:0x150000])]
    (out / "strings.txt").write_text("\n".join(texts) + "\n")
    print(f"Saved {len(data) // 1024} KB to {out.relative_to(ROOT)}/. The board was reset: press START to run it.")
    print("Run `iro backup` to send it to GitHub.")
    return 0


def cmd_restore(cfg, image):
    p = port(cfg)
    if input(f"Overwrite the whole PRIZM Pro flash with {image}? Type yes: ").strip() != "yes":
        return 1
    r = esptool("--port", p, "-b", "921600", "write-flash", "0", image)
    print((r.stdout + r.stderr)[-800:])
    return r.returncode


def main():
    args = sys.argv[1:] or ["ping"]
    cfg = load_config()
    if cfg["devices"]["base_controller"]["usb"] not in usb_ids():
        print("PRIZM Pro not on USB.")
        return 1
    if args[0] == "ping":
        return cmd_ping(cfg)
    if args[0] == "backup":
        return cmd_backup(cfg)
    if args[0] == "restore" and len(args) == 2:
        return cmd_restore(cfg, args[1])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
