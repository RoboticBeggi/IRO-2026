# Recovery: if the Jetson or its SSD dies

The project must never exist only on the Jetson. Check the state any time with
`iro backup status`; prove a copy restores with `iro backup verify`.

## What is protected, and where

| What | Where it lives | Backed up by |
|---|---|---|
| Code, plans, configs, routines | this repo | `autosave/<hostname>` snapshot every 10 min → GitHub (online) + USB (when plugged in) |
| Uncommitted and untracked work | working tree | same snapshot (your branch is not touched) |
| SO-ARM101 calibration | `arm/calibration/` | in the repo |
| What each component remembers (servo registers, lidar, OAK-D calibration, firmware versions) | the devices | `iro hw snapshot` → `backup/hardware/` |
| PRIZM program | the PRIZM | `iro prizm backup` → `backup/prizm/` |
| Python packages | `~/.venvs/iro` | versions in `env/`; whole env on USB with `iro backup kit` |
| Arduino tools, arduino-cli | `~/.arduino15`, `~/.local/opt` | USB with `iro backup kit` |
| Claude Code memory notes | `~/.claude/...` | copied to `backup/claude-memory/` |
| Device names (`/dev/so101`, `/dev/rplidar`) | `tools/99-iro-usb.rules` | in the repo |
| Models, datasets, videos (>50 MB) | `models/`, `datasets/` | USB only (too big for GitHub) |

## Backups at the competition (no internet)

GitHub is out of reach at the venue, so the USB drive is the backup there.

- Before leaving: plug in the `IRO_BACKUP` drive and run `iro backup kit`
  (adds the Python environment and tools, so nothing needs downloading),
  then `iro backup verify`.
- At the venue: keep the drive plugged in. Every 10 minutes the backup writes a
  dated git bundle (newest 30 kept) and a plain copy. Offline is detected in
  3 seconds, so nothing hangs.
- Bring a **spare SSD with the whole system already installed** and tested
  (`iro doctor` all green on it). Swapping an SSD takes 5 minutes; reinstalling
  takes hours. This is the real competition failsafe.
- Rule check: ask the Secretariat whether a backup drive and pre-installed
  software are allowed (rules 2.3 and 6.1-6.3; see the open questions in the roadmap).

## One-time setup (per Jetson)

```bash
iro setup                      # or tools/bootstrap.sh before iro exists
tools/setup-push-key.sh        # add the printed key on GitHub with write access
iro backup && iro backup verify
```

## Recovery from zero, with internet

1. Flash JetPack 6 on the new SSD.
2. `git clone https://github.com/roboticbeggi/iro-2026.git ~/iro-2026 && cd ~/iro-2026`
3. If the snapshot is newer than `main`: `git checkout -b recovered origin/autosave/ubuntu`
4. `tools/bootstrap.sh`, log out and back in.
5. `tools/setup-push-key.sh` (the old key died with the SSD; delete it on GitHub).
6. `iro doctor`. Calibration comes back from the repo; no recalibration needed.

## Recovery from zero, offline (USB drive only)

1. Flash JetPack 6 (bring the flashed spare, or an SD card image).
2. Restore the code from the newest bundle:
   ```bash
   B=$(ls -1t /media/$USER/IRO_BACKUP/iro-2026-backup/bundles/*.bundle | head -1)
   git clone "$B" ~/iro-2026 && cd ~/iro-2026 && git checkout -b recovered origin/autosave/ubuntu
   ```
3. Restore the tools: copy `offline-kit/venv-iro` to `~/.venvs/iro`,
   `offline-kit/local-opt` to `~/.local/opt`, `offline-kit/arduino15` to `~/.arduino15`.
   Same user name (`orin_nano`) is required for the copied environment to work.
4. ROS 2 offline: `sudo dpkg -i offline-kit/apt-debs/*.deb` (only if they were
   cached), otherwise this step needs internet. Another reason for the spare SSD.
5. `ln -sf ~/iro-2026/tools/iro ~/.local/bin/iro && iro doctor`

## Habits

- `iro doctor` before every session; it also checks the backups.
- Commit milestones to `main` by hand; the autosave is the safety net.
- `iro backup` after anything you would hate to redo (calibration, a new routine).
- `iro hw snapshot` after changing any device setting.
- Practise the offline recovery once before December, and time it.
