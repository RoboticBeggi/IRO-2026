---
name: record-and-review
description: User rule - everything with a camera/feed/recordable hardware must record each run so Claude can review it visually
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-21T08:44:04.812Z
---

Every program that uses a camera, a live feed, or any hardware that can be recorded must record its runs (video + per-step log + events), and Claude reviews those recordings (contact sheets, pictures, CSV) before and after changes instead of guessing.

**Why:** the user asked for it after tracking and calibration failures (21 Oct 2026); reviewing the recordings is what found the real causes (VitTrack drifting onto a shadow, the gripper out of the OAK-D's view, an inconsistent jaw feature, a 2 GB SVD).

**How to apply:** use `iro.record.run()` (r.video / r.log / r.snap / r.note); anything shown via `iro.stream.feed().show()` is recorded automatically as feed.mp4. Runs go to ~/iro-runs/<time>_<program>/; `iro review [RUN]` makes contact sheets. New hardware tools: add a recording from the start, and save enough raw data (e.g. sightings.npz) to re-solve without re-running the hardware (`--from RUN`). Related: [[jetson-hardware-state]].
