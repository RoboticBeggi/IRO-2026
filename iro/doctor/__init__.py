"""The robot doctor: `iro doctor` runs a checkup and explains every problem it finds."""

# Import order = report order
from . import checks_jetson, checks_devices, checks_arm, checks_backup  # noqa: F401,I001  (registers checks)
from .core import CHECKS, FAIL, INFO, OK, SKIP, WARN, Result, print_report, run, worst  # noqa: F401
