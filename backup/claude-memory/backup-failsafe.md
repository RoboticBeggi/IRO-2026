---
name: backup-failsafe
description: User lost a whole project to a dead Jetson SSD; everything must be recoverable from GitHub
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-20T02:10:57.623Z
---

The user's previous project was wiped when the Jetson SSD died, and they had to restart from zero. They asked for a failsafe.

**Why:** losing work again would be catastrophic this close to IRO 2026.
**How to apply:** anything new (calibration, routines, configs, install steps) must live in the repo or be captured by tools/backup.sh (autosave/<hostname> branch → GitHub, optional USB mirror). New dependencies go into env/requirements*.txt and tools/bootstrap.sh. Large files go on USB, not GitHub. Remind the user to run backup after calibration or a newly taught routine. See RECOVERY.md and [[jetson-hardware-state]].
