---
name: arm-motion-tuning
description: "SO-101 motion speed-ups of 15 Nov 2026: polling settle, learnt sag model, move_line, P 24, speed 1.5 — measured numbers"
metadata:
  node_type: memory
  type: project
  originSessionId: 7d60c8c7-5061-4fb7-a308-470ec7e19543
  modified: 2026-11-15T07:00:16.541Z
---

15 Nov 2026 (user asked for smoother, "instant" arm moves; done autonomously, tested on hardware):
- iro/motion.py `_settle` polls positions every 10 ms (6-servo read = 1.0 ms) instead of fixed 0.25 s waits; nudges only joints still > 3 counts off (nudging all made moves hunt). Settle now 0.1-0.2 s (was 0.5-1 s).
- `SagModel` (arm/calibration/sag_model.json, backed up): learns goal offset = a*gravity lever + b*approach direction per joint, used after 8 moves. Learnt at P 24; the P 16 version is kept as sag_model.p16.json. Delete/reset it if P gains change.
- config/robot.yaml `devices.arm.p_gain: {shoulder_lift: 24, elbow_flex: 24}` written to servo RAM on every Arm connect. Measured: P 24 sag 38->24 counts (shoulder), 24->15 (elbow), no shake; P 32 made wrist_flex shake. Wrist stays 16.
- `pick.max_joint_speed` 0.8 -> 1.5 rad/s (1.6 tested: same accuracy, no extra heat, 8-10 cm moves 1.13 -> 0.68 s).
- `Arm.move_line()`: straight-line, one minimum-jerk move, IK precomputed every 5 mm (iters=40: ik_down at reach edge runs all 200 iterations, ~280 ms each), `watch` callback per tick. grasp.descend uses it: fast 15 cm/s above object top+2 cm, careful 4 cm/s with blockage = tip lags command > 8 mm beyond its free-air lag (free-air lag ~3 mm). Descent 22.5 s (orig) -> ~5 s.
- Open: wrist-camera Tape matcher accepted a white block (match 0.25 vs tape 0.28-0.30) — thin margin on the mono camera; jaw direction for a standing tape still comes from the oblique wrist outline (OAK-D footprint fix proposed, not built).

Related: [[jetson-hardware-state]], [[record-and-review]].
