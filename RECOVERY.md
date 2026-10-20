# Recovery: if the Jetson or its SSD dies

The project must never exist only on the Jetson. Everything needed to rebuild
it is pushed to GitHub automatically.

## What is protected, and where

| What | Where it lives | Backed up by |
|---|---|---|
| Code, plans, configs, keyframe routines | this repo | git, and an `autosave/<hostname>` snapshot every 30 min |
| Uncommitted and untracked work | working tree | `autosave/<hostname>` snapshot (your branch is not touched) |
| SO-ARM101 calibration | `arm/calibration/` (via `env.sh`) | git |
| Motor IDs | written in the servos themselves | survive SSD loss; listed in `arm/README.md` |
| Python packages, exact versions | `env/requirements*.txt` | git; rebuilt by `tools/bootstrap.sh` |
| Claude Code memory notes | `~/.claude/...` | copied to `backup/claude-memory/` |
| Device names (`/dev/so101`, `/dev/rplidar`) | `tools/99-iro-usb.rules` | git |
| Large files: models, datasets, videos (>50 MB) | `models/`, `datasets/` | **not GitHub.** USB drive labelled `IRO_BACKUP` (mirrored automatically when plugged in) |

Rebuildable and not backed up: `~/.venvs/iro`, ROS 2 build folders.

## One-time setup (per Jetson)

```bash
tools/setup-push-key.sh        # then add the printed key on GitHub with write access
tools/backup.sh                # test: should end with "pushed to GitHub"
tools/install-autobackup.sh    # every 30 min + after each boot
```

Check that it is running: `tail ~/.iro-backup.log`

## Recovery from zero

1. Flash JetPack 6 on the new SSD (NVIDIA SDK Manager or the SD card image).
2. Get the code back:
   ```bash
   git clone https://github.com/roboticbeggi/iro-2026.git ~/iro-2026
   cd ~/iro-2026
   git log --oneline -3 origin/autosave/$(hostname)   # newest snapshot, if newer than main
   ```
   If the snapshot is newer than `main`, continue from it:
   `git checkout -b recovered origin/autosave/<old-hostname>`
3. Rebuild the software: `tools/bootstrap.sh`, then log out and back in.
4. Turn backups back on: `tools/setup-push-key.sh` (new key, the old one died
   with the SSD; delete the old one on GitHub), then `tools/install-autobackup.sh`.
5. Check the arm: `source env.sh && python arm/scan_motors.py`.
   The calibration is restored from the repo, so the arm works without recalibrating.
6. Copy `models/` and `datasets/` back from the USB drive.

## Habits

- Commit real milestones to `main` by hand with a clear message; the autosave
  is the safety net, not the history.
- Run `tools/backup.sh` by hand after anything you would hate to redo
  (calibration, a newly taught routine).
- Plug the `IRO_BACKUP` USB drive in at least once a week.
- Practise this recovery once before the competition on a spare SSD or SD
  card, and time it.
