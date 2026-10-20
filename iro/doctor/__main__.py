"""iro doctor [full|quick|watch] [group ...] [--json] [--every N]

  full   (default) every check, including slow ones (GPU, SSD log)
  quick  skip the slow checks
  watch  repeat the risk checks (power, heat, servos) every few seconds and
         shout when something changes; Ctrl+C to stop
  group  only these groups: jetson devices arm backup
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from . import FAIL, OK, WARN, print_report, run, worst

LOG = Path.home() / ".iro-doctor.log"


def watch(groups, every):
    print(f"Watching power, heat and servos every {every}s. Ctrl+C to stop. Log: {LOG}")
    last = {}
    try:
        while True:
            results = run(groups, mode="watch")
            stamp = time.strftime("%H:%M:%S")
            for r in results:
                key = (r.group, r.name)
                if last.get(key) != r.status:
                    if r.status in (WARN, FAIL) or key in last:
                        line = f"{stamp} {r.status:<4} {r.group}/{r.name}: {r.detail}"
                        print(("\a" if r.status == FAIL else "") + line + (f"\n         → {r.fix}" if r.fix and r.status != OK else ""))
                        with LOG.open("a") as f:
                            f.write(line + "\n")
                    last[key] = r.status
            summary = " | ".join(f"{r.name} {r.detail.split(';')[0]}" for r in results if r.name in ("power supply", "temperature", "servo power"))
            print(f"\r{stamp} {summary[:150]}", end="", flush=True)
            time.sleep(every)
    except KeyboardInterrupt:
        print()


def main():
    p = argparse.ArgumentParser(prog="iro doctor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="full", choices=["full", "quick", "watch"])
    p.add_argument("groups", nargs="*", help="jetson devices arm backup")
    p.add_argument("--json", action="store_true")
    p.add_argument("--every", type=float, default=3.0)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    if a.mode == "watch":
        watch(a.groups or None, a.every)
        return 0
    results = run(a.groups or None, mode=a.mode)
    if a.json:
        print(json.dumps([dataclasses.asdict(r) for r in results], default=str, indent=1))
    else:
        print_report(results, color=sys.stdout.isatty(), verbose=a.verbose)
    return worst(results)


if __name__ == "__main__":
    sys.exit(main())
