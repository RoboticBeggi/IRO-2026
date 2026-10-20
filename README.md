# IRO 2026: heritage storyteller robot

IRO 2026 Creative Category (Austria). Plan: [PLAN.md](PLAN.md). Roadmap: the `.docx`.

## Everyday commands

Everything goes through one command, from any folder. `iro help` lists them all.

```bash
iro doctor            # checkup: what is wrong, why, and how to fix it
iro doctor watch      # live watch of power, heat and servos while you work
iro up                # start the robot in ROS 2 (everything plugged in)
iro mirror            # RViz arm that follows the real arm
iro arm scan          # are the 6 servos OK?
iro backup            # back up now (also runs by itself every 10 min)
```

## Layout

| Folder | What |
|---|---|
| `config/robot.yaml` | every device, port and safety limit, in one place |
| `iro/` | Python: doctor, hardware helpers, PRIZM and snapshot tools |
| `ros2_ws/src/iro_ros` | ROS 2 nodes: `arm_node`, `doctor_node` |
| `ros2_ws/src/iro_bringup` | launch files, SO-101 model (URDF), RViz layout |
| `arm/` | arm calibration and notes |
| `tools/` | `iro` command, setup, backup |
| `backup/` | copies of things that live outside the repo, hardware snapshots |
| `salvage/` | what was recovered from the lost PUSAKA project |

If the Jetson or its SSD dies: [RECOVERY.md](RECOVERY.md).
