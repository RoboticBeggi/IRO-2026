---
name: jetson-hardware-state
description: Actual devices and software found on the Orin Nano dev box as of 2026-10-20 (differs from roadmap doc in places)
metadata:
  node_type: memory
  type: project
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-20T02:01:55.806Z
---

Sweep on 2026-10-20 of the Jetson Orin Nano (JetPack R36.5, Ubuntu 22.04, 7.4 GiB RAM, CUDA 12.6, 467G NVMe):
- ROS 2 NOT installed natively (no /opt/ros); docker present but user not in docker group.
- USB: reSpeaker XVF3800 4-mic array (roadmap calls it "Cytron" — likely the reseller), Intel MyriadX = Luxonis OAK-D (the "depth camera"), CP210x on /dev/ttyUSB0 (likely RPLidar), CH343 serial (likely SO-ARM101 Feetech bus board, /dev/ttyACM0), "openaicam" (likely the global-shutter arm camera), /dev/video0-1.
- ~/jps holds Jetson Platform Services bundles: vlm, zero_shot_detection, ai_nvr, nvstreamer.
- Repo iro-2026 holds only README + roadmap .docx (no code yet).

**Why:** 8 GB shared RAM must fit LLM + STT + TTS + vision; device mapping guesses need confirming.
**How to apply:** verify device roles before writing drivers/launch files; size models to RAM. See [[iro-2026-roadmap]].
