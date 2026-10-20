"""Read-only access to the SO-ARM101 servo bus (Feetech STS3215) through LeRobot."""

from contextlib import contextmanager

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

STS3215_MODEL = 777

# Servo "Status" register (address 65) error bits, from the Feetech STS manual
STATUS_BITS = {
    0: "voltage out of range",
    1: "position sensor fault",
    2: "overheated",
    3: "over-current",
    5: "overloaded",
}


def decode_status(value: int) -> list[str]:
    return [text for bit, text in STATUS_BITS.items() if value >> bit & 1]


@contextmanager
def open_bus(port: str, servo_ids: dict[str, int]):
    """Connect without handshake and never touch torque on the way out."""
    bus = FeetechMotorsBus(
        port=port,
        motors={name: Motor(id_, "sts3215", MotorNormMode.RANGE_M100_100) for name, id_ in servo_ids.items()},
    )
    bus.connect(handshake=False)
    try:
        yield bus
    finally:
        bus.disconnect(disable_torque=False)


def read_health(bus) -> dict[str, dict]:
    """Per joint: raw position, temperature, volts, load %, status errors, torque on/off."""
    reads = {
        reg: bus.sync_read(reg, normalize=False)
        for reg in ("Present_Position", "Present_Temperature", "Present_Voltage", "Present_Load", "Status", "Torque_Enable")
    }
    return {
        name: {
            "pos": reads["Present_Position"][name],
            "temp_c": reads["Present_Temperature"][name],
            "volts": reads["Present_Voltage"][name] / 10,
            "load_pct": abs(reads["Present_Load"][name]) / 10,
            "errors": decode_status(reads["Status"][name]),
            "torque": bool(reads["Torque_Enable"][name]),
        }
        for name in bus.motors
    }
