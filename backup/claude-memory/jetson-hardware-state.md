---
name: jetson-hardware-state
description: "Devices, software env and setup quirks on the Orin Nano dev box (as of 2026-10-20)"
metadata:
  node_type: memory
  type: project
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-20T02:10:56.520Z
---

Jetson Orin Nano: JetPack R36.5, Ubuntu 22.04, 7.4 GiB RAM, CUDA 12.6, hostname `ubuntu`.
- USB: reSpeaker XVF3800 mic array, OAK-D (MyriadX) depth cam, RPLidar on CP2102N (/dev/ttyUSB0, serial 14ec6b...), SO-ARM101 Feetech board CH343 (/dev/ttyACM0, serial 5B91044420), "openaicam" arm camera.
- Python env: ~/.venvs/iro made with uv (python3.10-venv not installed, no sudo). torch 2.10.0 from pypi.jetson-ai-lab.io/jp6/cu126 needs libcudss.so.0 → pip nvidia-cudss-cu12, LD_LIBRARY_PATH set in repo env.sh. LeRobot 0.4.4.
- ROS 2 not installed yet. User not in dialout/docker groups yet (needs their sudo password).
- SO-101: follower only; servo IDs 1-6 were already set in the earlier (lost) project. Calibration lives in repo arm/calibration via HF_LEROBOT_CALIBRATION.

**Why:** a previous SSD death wiped the user's whole project, so recoverability matters most.
**How to apply:** keep all state in the repo or backup/; see [[backup-failsafe]].
