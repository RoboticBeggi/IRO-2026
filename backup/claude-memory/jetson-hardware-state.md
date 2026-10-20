---
name: jetson-hardware-state
description: "Devices, software env, setup quirks and decisions on the Orin Nano robot (as of 2026-10-20)"
metadata:
  node_type: memory
  type: project
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-20T03:10:52.761Z
---

Jetson Orin Nano: JetPack R36.5, Ubuntu 22.04, 7.4 GiB RAM, CUDA 12.6, hostname `ubuntu`, MAXN_SUPER.
- Devices (all in repo config/robot.yaml): SO-101 follower on /dev/so101 (12 V STS3215, fw 3.10, IDs 1-6 from old project); RPLidar **S2 at 1,000,000 baud** on /dev/rplidar (latched health error code 4 cleared by reset 0xA5 0x40); openaicam wrist cam /dev/video0; OAK-D (needs udev 03e7); reSpeaker XVF3800 fw 2.0.6; PRIZM not yet connected (as of 2026-10-20).
- Python env ~/.venvs/iro (uv): Jetson torch 2.10 + nvidia-cudss-cu12 (LD_LIBRARY_PATH in env.sh), LeRobot 0.4.4, depthai 3.11. **numpy pinned <2 and opencv-headless <4.11** so ROS 2 Humble message libs load; accepted conflict with rerun-sdk.
- arduino-cli + avrdude 8.0 in ~/.local/opt and ~/.arduino15 (no sudo).
- ROS 2 Humble installed 2026-10-20. Tested live: arm_node /joint_states 30 Hz + /arm/torque, doctor_node /diagnostics, rplidar_node (DenseBoost 10 Hz), own camera_node 29 Hz. Not yet tested: OAK-D (depth part), RViz on screen, joint_signs.
- Gotchas: ROS python nodes must be plain scripts with `#!/usr/bin/env python3` (console_scripts get /usr/bin/python3, no venv). Wrist cam is MJPG-only → v4l2_camera can't use it. rclpy Image.data must be array('B') (plain bytes = 149 ms/frame).
- Open problem: 20 Oct calibration has no limits on elbow_flex, wrist_flex (range 0-4095); user must recalibrate.
- Old lost project was named PUSAKA; see repo salvage/README.md.

**Why:** a previous SSD death wiped the user's whole project, so recoverability matters most.
**How to apply:** everything is run via the `iro` command (tools/iro); keep state in the repo; see [[backup-failsafe]].
