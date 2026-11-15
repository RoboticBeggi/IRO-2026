---
name: arm-motion-tuning
description: "SO-101 speed/accuracy work of 15 Nov 2026: settle, sag model, move_line, LM IK, OAK-D scout-guided pick, measured limits and open items"
metadata:
  node_type: memory
  type: project
  originSessionId: 7d60c8c7-5061-4fb7-a308-470ec7e19543
  modified: 2026-11-15T08:32:46.174Z
---

15 Nov 2026, user asked for smooth, "instant" detect-centre-pick and gave free rein to experiment. Full write-up for the team: arm/ARM_NOTES.md (limits table, maths, ideas). Key facts not obvious from the code:

**Motion (iro/motion.py, iro/kinematics.py)**
- `_settle` polls every 10 ms; arrived = within 3 counts AND still for 3 reads (in-tolerance alone returned while joints crept up to 8 counts for 0.4 s). `wait_still()` before pictures.
- `SagModel` (arm/calibration/sag_model.json): learnt goal offsets, learnt at P 24 (sag_model.p16.json = old). Reset if P gains change.
- config: `devices.arm.p_gain {shoulder_lift: 24, elbow_flex: 24}` (P 32 shook wrist_flex), `pick.max_joint_speed` 1.5.
- `Arm.move_line()` straight minimum-jerk line with a `watch` callback.
- `ik_down` is now Levenberg-Marquardt (6.5 ms reachable, old ~110 ms; never meaningfully worse over 500-700 random targets). Callers that only probe reachability pass `iters<100` to skip the slow old-solver fallback.

**Picture timing (the cause of slow/erratic centring)**
- Floor-texture Jacobian (`floor_jacobian`) is noise on the plain wooden floor (7 tries differing 3x and in sign). Replaced by `config/wrist_camera.yaml` (Jg 2x2 in the gripper frame + Mz), measured on the object by `iro pick calibrate NAME`; fit 4.5 px.
- Encoders read "still" while the arm/camera still wobble: `DownEye.still_frame()` waits until two frames agree (a picture was 28 px off with joints within 3 counts).
- Wrist camera is oblique: 4941 px/m sideways, 2107 px/m along the view, 1093 px/m per metre of height.
- `core_centre` (thick part of the SAM mask) replaces the mask mean.

**OAK-D scout (iro/oakguide.py)**
- Arm at look height (15 cm) is above the OAK-D's field of view, so the scout can look any time. View starts 18 cm from the arm base; depth valid from ~11 cm from the lens.
- Low viewpoint sees only the near face: `complete()` uses the three taught sizes to get pose + true centre.
- OAK-D mount shifted since the 21 Oct calibration (3-4.5 cm off, ~7 deg). `Correction` learns a rigid fix from each successful guided pick (config/scout_pairs.yaml; emptied when efad_calibration.yaml's date changes). Aim went 84-114 px -> 12-33 px. Proper fix still open: re-run `iro efad calibrate` on a clear floor.
- Scout's IR dots are visible in the wrist camera picture.

**Pick (iro/grasp.py)**
- Guided path: scout -> `roll_across` -> look pose -> `centre2` -> descend -> close_on from FULLY open (starting 12 mm from the object made the contact detector miss a real grip) -> lift -> OAK floor check -> save pair. Fallback = old wrist search (`IRO_NO_SCOUT=1` forces it; verified working).
- Blockage = tip stalled while command descends AND tip > 5 mm above command (a pure lag threshold false-alarmed at 33 cm reach: free-air lag 8-13 mm).
- `--to=X,Y` pick-and-place, tools/pick_loop.sh.
- Fast put-down toppled a tape standing on its rim; gentler version written, NOT yet verified. Loop beyond first pick NOT yet verified (tape kept ending up lying flat).

**Measured limits**: reach at grasp height 30.8 cm straight down, 40 cm at 45 deg lean (hardware-checked 28-40 cm); 22 cm highest straight-down; gripper 71.4 mm max, ~55 mm object with 8 mm margins (lying tape 70 mm: not graspable from above); look-pose repeatability +-3 mm (worst 5.9).

**Jetson power mode**: was 15W during all of this (comes up that way after reboot; needs `sudo nvpmodel -m 4`, no passwordless sudo). SAM outline 0.77 s, pick start-up 16-19 s at 15W.

The user actively resets the tape during tests (stands it up / moves it), so scene state can change between runs: re-scout before assuming.

Related: [[jetson-hardware-state]], [[record-and-review]].
