#!/usr/bin/python3
"""Diagnose the Waveshare UPS Power Module (C) under the Orin NX.

Logs the module's INA219 (I2C bus 7, 0x41) and the Jetson's own input rail
(INA3221 VDD_IN) at ~20 Hz, walks you through BOOT / adapter tests, then
prints a verdict. Every sample is written + fsynced to ~/iro-runs/ups/, so if
the Jetson loses power mid-test the log survives: re-check it with
    tools/ups_check.py analyze ~/iro-runs/ups/<file>.csv

Run with the SYSTEM python (the iro venv has no smbus):
    /usr/bin/python3 tools/ups_check.py            # guided test
    /usr/bin/python3 tools/ups_check.py watch      # live readout only
"""
import csv, glob, os, statistics as st, sys, time

BUS, ADDR, RSHUNT = 7, 0x41, 0.01          # 0.01 ohm per Waveshare demo
RUNDIR = os.path.expanduser("~/iro-runs/ups")


class Ina219:
    def __init__(self):
        import smbus
        self.b = smbus.SMBus(BUS)
        # 16 V range, /2 gain (80 mV), 12-bit x2 averaging both, continuous
        cfg = (0 << 13) | (1 << 11) | (0x9 << 7) | (0x9 << 3) | 7
        self._w(0x00, cfg)

    def _w(self, reg, v):
        self.b.write_i2c_block_data(ADDR, reg, [(v >> 8) & 0xFF, v & 0xFF])

    def _r(self, reg):
        d = self.b.read_i2c_block_data(ADDR, reg, 2)
        v = (d[0] << 8) | d[1]
        return v - 65536 if v > 32767 else v

    def read(self):
        vbus = (self._r(0x02) & 0xFFFF) >> 3
        return vbus * 0.004, self._r(0x01) * 10e-6 / RSHUNT   # V, A


def jetson_in():
    """Jetson VDD_IN (V, A) from the on-module INA3221."""
    for lab in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
        if open(lab).read().strip() == "VDD_IN":
            n = lab[:-6].rsplit("in", 1)[1]
            d = os.path.dirname(lab)
            v = int(open(f"{d}/in{n}_input").read()) / 1000
            a = int(open(f"{d}/curr{n}_input").read()) / 1000
            return v, a
    return float("nan"), float("nan")


PHASES = [
    ("baseline", 8, "Leave everything as it is now. Don't touch anything."),
    ("boot", 15, "Press the BOOT button on the UPS module ONCE, now."),
    ("unplug", 20, "Pull the adapter plug OUT now (if the Jetson dies, that is a result too:\n"
                   "  power back up and run:  tools/ups_check.py analyze <log>)"),
    ("replug", 15, "Plug the adapter back into the UPS MODULE's jack now."),
]


def guided():
    os.makedirs(RUNDIR, exist_ok=True)
    path = f"{RUNDIR}/{time.strftime('%Y%m%d-%H%M%S')}.csv"
    ina = Ina219()
    f = open(path, "w", buffering=1)
    w = csv.writer(f)
    w.writerow(["t", "phase", "batt_v", "batt_a", "jet_v", "jet_a"])
    print(f"logging to {path}\n")
    t0 = time.time()
    last_sync = 0
    for name, secs, msg in PHASES:
        print(f"== {name.upper()} ({secs}s) ==\n  {msg}")
        if name != "baseline":
            input("  press Enter when ready, then do it... ")
        end = time.time() + secs
        while time.time() < end:
            bv, ba = ina.read()
            jv, ja = jetson_in()
            w.writerow([f"{time.time()-t0:.2f}", name, f"{bv:.3f}", f"{ba:.4f}",
                        f"{jv:.3f}", f"{ja:.3f}"])
            if time.time() - last_sync > 0.5:
                f.flush(); os.fsync(f.fileno()); last_sync = time.time()
                print(f"\r  batt {bv:6.2f} V {ba:+6.2f} A | jetson in {jv:5.2f} V {ja:4.2f} A ",
                      end="", flush=True)
            time.sleep(0.05)
        print()
    f.close()
    analyze(path)


def watch():
    ina = Ina219()
    while True:
        bv, ba = ina.read()
        jv, ja = jetson_in()
        print(f"{time.strftime('%T')}  batt {bv:6.2f} V {ba:+6.2f} A | jetson in {jv:5.2f} V {ja:4.2f} A")
        time.sleep(0.5)


def analyze(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    print(f"\n== RESULTS ({path}) ==")
    stats = {}
    for name, _, _ in PHASES:
        rs = by.get(name)
        if not rs:
            print(f"{name:9s} no data (Jetson lost power during/before this phase?)")
            continue
        bv = [float(r["batt_v"]) for r in rs]; ba = [float(r["batt_a"]) for r in rs]
        jv = [float(r["jet_v"]) for r in rs]
        s = dict(v=st.mean(bv), vmin=min(bv), vmax=max(bv), vsd=st.pstdev(bv),
                 a=st.mean(ba), amin=min(ba), amax=max(ba), jmin=min(jv), jmax=max(jv), n=len(rs))
        stats[name] = s
        print(f"{name:9s} batt {s['v']:5.2f} V (min {s['vmin']:5.2f} max {s['vmax']:5.2f} sd {s['vsd']:.3f})"
              f"  I {s['a']:+5.2f} A ({s['amin']:+5.2f}..{s['amax']:+5.2f})"
              f"  jetson {s['jmin']:5.2f}-{s['jmax']:5.2f} V  n={s['n']}")

    print("\n== VERDICT ==")
    b = stats.get("baseline")
    if b:
        if b["vsd"] > 0.08 and abs(b["a"]) < 0.03:
            print("- Baseline voltage jumps around with no current: the INA219 looks at a\n"
                  "  FLOATING node, i.e. the module's output/battery path is switched OFF\n"
                  "  (protection chip latched). A real battery reads steady to ~0.01 V.")
        elif abs(b["a"]) < 0.03:
            print("- Baseline: battery steady, no current. Idle/full (normal if adapter feeds the Jetson).")
        elif b["a"] < 0:
            print("- Baseline: battery is DISCHARGING -> Jetson is running on the battery.")
        else:
            print("- Baseline: battery is CHARGING.")
        if 9 <= b["v"] <= 12.6 and b["vsd"] < 0.08:
            print(f"  pack {b['v']:.2f} V = {b['v']/3:.2f} V/cell average (one weak cell can still hide in this).")
    bo = stats.get("boot")
    if bo and b:
        if abs(bo["amax"]) > 0.1 or abs(bo["amin"]) > 0.1:
            print(f"- BOOT: current flowed ({bo['amin']:+.2f}..{bo['amax']:+.2f} A)", end="")
            if abs(bo["a"]) < 0.05:
                print(" but then stopped -> output turns on and the protection trips again.")
            else:
                print(" and kept flowing -> BOOT enabled the output.")
            sag = b["v"] - bo["vmin"]
            if sag > 0.6:
                print(f"  Voltage sagged {sag:.2f} V under load -> weak/high-resistance cell(s). Measure each cell.")
        elif bo["vsd"] < b["vsd"] / 2 and b["vsd"] > 0.08:
            print("- BOOT: voltage went from floating to steady -> battery path connected (no load drawing).")
        else:
            print("- BOOT: no change at all seen by the INA219 -> BOOT not activating the protection chip.\n"
                  "  Check cells (orientation, each 3.0-4.2 V, all within 0.1 V) and the WARNING LED.")
    u = stats.get("unplug")
    if not u:
        print("- UNPLUG: no data -> the Jetson died when the adapter was pulled. The module is\n"
              "  NOT carrying the Jetson: output off (see above) or pogo pins not contacting.")
    else:
        if u["jmin"] < 13.5:
            print(f"- UNPLUG: Jetson survived on battery (VDD_IN fell to {u['jmin']:.2f} V). UPS WORKS.")
            if u["amin"] < -0.2:
                print(f"  Battery delivered up to {-u['amin']:.2f} A.")
        elif u["jmin"] > 17:
            print("- UNPLUG: Jetson input never dropped -> adapter probably wasn't pulled, or it feeds\n"
                  "  the Jetson's own jack instead of the module. Feed the MODULE's jack only.")
    r = stats.get("replug")
    if r and r["amax"] > 0.1:
        print(f"- REPLUG: battery charging at up to {r['amax']:.2f} A. Charger path works.")
    elif r and b and b["v"] >= 12.0:
        print("- REPLUG: no charge current; expected while pack >= 12.0 V (module only recharges below 12 V).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "watch":
        watch()
    elif cmd == "analyze":
        analyze(sys.argv[2])
    else:
        guided()
