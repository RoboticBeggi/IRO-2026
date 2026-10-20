# SO-ARM101 follower arm

Follower arm only (no leader). Set up and calibrated with LeRobot 0.4.4.
Routines are taught by hand and replayed as keyframes (see `PLAN.md`, Tier 1 item 4).

| Joint | Servo ID | Model |
|---|---|---|
| shoulder_pan | 1 | STS3215 |
| shoulder_lift | 2 | STS3215 |
| elbow_flex | 3 | STS3215 |
| wrist_flex | 4 | STS3215 |
| wrist_roll | 5 | STS3215 |
| gripper | 6 | STS3215 |

IDs are stored in the servos themselves (set in the earlier project), so they
survive an SSD failure. Only re-run `lerobot-setup-motors` when a servo is
replaced; it needs one servo connected at a time.

## Every session

```bash
iro arm scan          # read-only: all 6 IDs answer, temps, voltages
iro doctor arm        # plus calibration sanity and servo error flags
iro mirror            # RViz model follows the real arm
```

Servos are the **12 V STS3215** version (they read about 11.6 V).

## Calibrate

Needed once after assembly, and again whenever the arm is rebuilt (so on
competition day too: practise it and time it).

```bash
iro arm calibrate     # runs lerobot-calibrate, then checks the result and backs it up
```

1. Torque turns off. Move every joint by hand to the **middle of its range**
   (arm roughly in an upright "L": upper arm vertical, forearm horizontal,
   wrist straight, gripper half open), then press Enter. This writes a homing
   offset into each servo.
2. Move **each joint slowly through its full range**, end to end, one at a
   time. Wrist roll is skipped (it turns all the way round). Press Enter when done.
3. It saves to `arm/calibration/robots/so_follower/iro_follower.json`.

`iro doctor arm` must then show no "no joint limits recorded" failure. A joint
with range 0-4095 (other than wrist_roll) was not moved end to end, or its
reading wrapped past the end because the middle pose was off; recalibrate.
The 20 Oct calibration has this problem on elbow_flex and wrist_flex.

The JSON and the servos' homing offsets must match. If you restore an old JSON
onto servos that were recalibrated since, LeRobot asks to recalibrate: press
Enter to write the file's values back to the servos, or `c` to recalibrate.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not connect on port` | Arm power on? USB in? `groups` shows `dialout`? (`tools/bootstrap.sh` adds it; log out and in) |
| `/dev/so101` missing | udev rule not installed: `tools/bootstrap.sh`, or use `--port /dev/ttyACM0` |
| An ID is missing | Loose 3-pin cable between servos; check the daisy chain from the board outward |
| Unexpected ID (e.g. 1 twice, or id 7) | Two servos share an ID: connect them one at a time and run `lerobot-setup-motors` |
| Servo `HOT` (55 C or more) | Rest pose with torque off; the shoulder suffers most when holding the puppet |
| Voltage well under the supply rating (5 V or 12 V, depending on the servo version) | Weak or wrong power supply, or a loose barrel plug |
