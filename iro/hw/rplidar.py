"""Minimal RPLidar protocol: device info and health, without spinning the motor."""

import time

import serial

MODELS = {0x71: "S2"}  # confirmed on our unit by its boot message; others show as hex
HEALTH = {0: "good", 1: "warning", 2: "error"}


def _request(s: serial.Serial, cmd: int, payload_len: int) -> bytes:
    s.reset_input_buffer()
    s.write(bytes([0xA5, cmd]))
    resp = s.read(7 + payload_len)
    if len(resp) != 7 + payload_len or resp[:2] != b"\xa5\x5a":
        raise TimeoutError(f"no valid reply to command 0x{cmd:02x} (got {resp.hex() or 'nothing'})")
    return resp[7:]


def reset(port: str, baud: int) -> str:
    """Restart the lidar's core (clears a latched error). Returns its boot message."""
    with serial.Serial(port, baud, timeout=1.0) as s:
        s.setDTR(False)
        s.write(b"\xa5\x40")
        time.sleep(1.0)
        return s.read(200).decode(errors="replace").strip()


def probe(port: str, baud: int) -> dict:
    """Return model, firmware, hardware, serial and health. Raises on no reply."""
    with serial.Serial(port, baud, timeout=0.5) as s:
        s.setDTR(False)  # DTR high stops the motor on most models; keep it as is
        time.sleep(0.05)
        s.write(b"\xa5\x25")  # STOP, in case another program left it scanning
        time.sleep(0.05)
        info = _request(s, 0x50, 20)
        health = _request(s, 0x52, 3)
    return {
        "model": MODELS.get(info[0], f"0x{info[0]:02x}"),
        "firmware": f"{info[2]}.{info[1]:02d}",
        "hardware": info[3],
        "serial": info[4:].hex().upper(),
        "health": HEALTH.get(health[0], f"unknown {health[0]}"),
        "error_code": health[1] | health[2] << 8,
    }
