# Salvage from the lost PUSAKA project

Searched on 20 Oct 2026 after the old Jetson's SSD died. What survived, and where it went.

| Where | Found | Used for |
|---|---|---|
| Claude Code session (started 8 Sep 2026 on the old Jetson, moved here 20 Oct) | Project name **PUSAKA**, path `~/Robotics/pusaka/ros2_ws`, packages `pusaka_description` (URDF with the official `so_arm101/so101_new_calib.urdf`, RViz config, `display.launch.py`), `pusaka_bringup` (`joint_state_pub` mirror node, 15 Hz), a `pusaka` Python module | Full `display.launch.py` recovered: [pusaka/display.launch.py](pusaka/display.launch.py). Rebuilt as `iro mirror` (`ros2_ws/src/iro_bringup/launch/mirror.launch.py` + `iro_ros/arm_node.py`) with the same official SO-101 URDF |
| SO-101 servos (EEPROM) | IDs 1-6 set by the old project, firmware 3.10. Homing offsets and limits were already overwritten by the 20 Oct recalibration | Full register dump: [servo_eeprom_2026-10-20.json](servo_eeprom_2026-10-20.json); now repeated by `iro hw snapshot` |
| RPLidar | Model S2, firmware 1.02, 1 Mbaud; nothing project-specific stored | `config/robot.yaml`, doctor lidar check |
| reSpeaker XVF3800 | Firmware 2.0.6; saved settings not readable until the udev rule is installed (`iro setup`) | To do: dump its parameters after setup |
| OAK-D | Factory calibration in its EEPROM; not readable until the udev rule is installed | `iro hw snapshot` saves it after `iro setup` |
| TETRIX PRIZM | **Not checked yet: not connected.** Holds the old base program, compiled (no source) | `iro prizm backup` copies the program + EEPROM and lists the text inside it |
| GitHub `roboticbeggi`, Hugging Face | Only this repo; no datasets, models or gists | - |
| This Jetson | Fresh install from 18 Jun; nothing from PUSAKA. An empty `~/.git` (no commits) | - |

Not searched (worth checking): the old SSD itself. If only the Jetson board died
and the SSD still works, put it in a USB NVMe enclosure and plug it in here:
the whole PUSAKA folder may still be on it. Also any laptop that edited or
uploaded to the PRIZM (Arduino IDE sketchbook folder), and phone photos of the
build.
