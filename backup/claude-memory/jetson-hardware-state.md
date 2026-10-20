---
name: jetson-hardware-state
description: "Devices, software env, setup quirks and decisions on the Orin Nano robot (as of 2026-10-20)"
metadata:
  node_type: memory
  type: project
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-20T04:36:57.265Z
---

Jetson Orin Nano: JetPack R36.5, Ubuntu 22.04, 7.4 GiB RAM, CUDA 12.6, hostname `ubuntu`, MAXN_SUPER.
- Devices (all in repo config/robot.yaml): SO-101 follower on /dev/so101 (12 V STS3215, fw 3.10, IDs 1-6 from old project); RPLidar **S2 at 1,000,000 baud** on /dev/rplidar (latched health error code 4 cleared by reset 0xA5 0x40); openaicam wrist cam /dev/video0; OAK-D (needs udev 03e7); reSpeaker XVF3800 fw 2.0.6; PRIZM not yet connected (as of 2026-10-20).
- Python env ~/.venvs/iro (uv): Jetson torch 2.10 + nvidia-cudss-cu12 (LD_LIBRARY_PATH in env.sh), LeRobot 0.4.4, depthai 3.11. **numpy pinned <2 and opencv-headless <4.11** so ROS 2 Humble message libs load; accepted conflict with rerun-sdk.
- arduino-cli + avrdude 8.0 in ~/.local/opt and ~/.arduino15 (no sudo).
- ROS 2 Humble installed 2026-10-20. Tested live: arm_node /joint_states 30 Hz + /arm/torque, doctor_node /diagnostics, rplidar_node (DenseBoost 10 Hz), own camera_node 29 Hz. Not yet tested: OAK-D (depth part), RViz on screen, joint_signs.
- Gotchas: ROS python nodes must be plain scripts with `#!/usr/bin/env python3` (console_scripts get /usr/bin/python3, no venv). Wrist cam is MJPG-only → v4l2_camera can't use it. rclpy Image.data must be array('B') (plain bytes = 149 ms/frame).
- Open problem: 20 Oct calibration has no limits on elbow_flex, wrist_flex (range 0-4095); user must recalibrate. iro.motion refuses to move until then.
- TETRIX PRIZM Pro = ESP32-S3 (303a:1001). Holds old PUSAKA base firmware (text protocol PING/MOVE vx vy wz/RAW/MOTOR/ENC/WDT, dump in backup/esp32s3/). Programs wait for green START after reset. Open its USB port with pyserial defaults (DTR/RTS on); setting them off resets the board.
- Arm recalibrated 20 Oct 12:00 (good, no wrap). Arm base mounted ~4 cm above the table on the robot body; URDF base_link origin = bottom of arm base, so table_z should be about -0.04. First teach gave table -7.1 cm / grasp below table → suspect reversed joint_signs; `iro arm check` wizard written to measure them.
- 20 Oct: arm check found all joint_signs +1. User saw RViz gripper turned sideways vs real → wrist_roll zero ~90 deg off (LeRobot middle pose != URDF zero). Arm setup order now: calibrate → check → `iro arm align` (live RViz offsets, roll first) → `iro arm zero` (table touches, LM solver, ±10 deg, leave-one-out outliers, touches logged to arm/calibration/touches_*.json; re-solve with `iro arm zero FILE`) → `iro pick teach`. First 42-touch zero run was lost (old tool didn't log) and failed because roll was 90 off.
- Arm pick pipeline (iro/kinematics.py, iro/motion.py, iro/pick.py; `iro pick`): YOLO11n 30 fps on mono wrist cam; FK matches ROS TF exactly. SO-101 can point straight down only up to ~8 cm (wrist limit 95 deg); look pose leans 25 deg outward at 12 cm. Visual servoing with runtime image Jacobian; `iro pick teach` sets table_z/grasp_z/grasp_pixel. Not yet run on hardware (waits for recalibration).
- Old lost project was named PUSAKA; see repo salvage/README.md.

**Why:** a previous SSD death wiped the user's whole project, so recoverability matters most.
**How to apply:** everything is run via the `iro` command (tools/iro); keep state in the repo; see [[backup-failsafe]].
