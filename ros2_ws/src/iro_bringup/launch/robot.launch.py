"""Start the robot. Normally run through `iro up [parts]`.

    ros2 launch iro_bringup robot.launch.py                   # everything plugged in
    ros2 launch iro_bringup robot.launch.py parts:=arm,rviz   # just these

Parts: arm (servos + robot model), lidar, camera (wrist), depth (OAK-D),
doctor (health on /diagnostics), rviz (needs a screen).
With parts:=all, devices that are not plugged in are skipped with a message,
so one missing cable never stops the whole robot.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from iro import load_config
from iro.hw.system import usb_ids

ALL = ["arm", "lidar", "camera", "depth", "doctor", "rviz"]


def static_tf(parent, child, xyz, rpy):
    return Node(package="tf2_ros", executable="static_transform_publisher", name=f"tf_{child}",
                arguments=["--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
                           "--roll", str(rpy[0]), "--pitch", str(rpy[1]), "--yaw", str(rpy[2]),
                           "--frame-id", parent, "--child-frame-id", child])


def plugged_in(dev) -> bool:
    ids = dev["usb"] if isinstance(dev["usb"], list) else [dev["usb"]]
    return bool(usb_ids() & set(ids))


def setup(context):
    cfg = load_config()
    dev, mounts = cfg["devices"], cfg.get("mounts", {})
    share = Path(get_package_share_directory("iro_bringup"))
    asked = context.launch_configurations["parts"]
    auto = asked == "all"
    parts = ALL if auto else [p.strip() for p in asked.split(",") if p.strip()]
    unknown = set(parts) - set(ALL)
    if unknown:
        raise RuntimeError(f"unknown parts {sorted(unknown)}; choose from {ALL}")

    actions = []

    def skip(part, why):
        actions.append(LogInfo(msg=f"[iro] {part} skipped: {why}"))

    if "arm" in parts:
        if auto and not plugged_in(dev["arm"]):
            skip("arm", "servo board not on USB")
        else:
            urdf = (share / "urdf" / "so101" / "so101_new_calib_camera.urdf").read_text()
            actions += [
                Node(package="robot_state_publisher", executable="robot_state_publisher",
                     parameters=[{"robot_description": urdf}], output="screen"),
                Node(package="iro_ros", executable="arm_node", output="screen"),
            ]

    if "lidar" in parts:
        if auto and not plugged_in(dev["lidar"]):
            skip("lidar", "not on USB")
        else:
            m = mounts.get("lidar", {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})
            actions += [
                Node(package="rplidar_ros", executable="rplidar_node", name="lidar", output="screen",
                     parameters=[{"serial_port": dev["lidar"]["port"], "serial_baudrate": dev["lidar"]["baud"],
                                  "frame_id": "laser", "angle_compensate": True}]),
                static_tf("base_link", "laser", m["xyz"], m["rpy"]),
            ]

    if "camera" in parts:
        if auto and not plugged_in(dev["arm_camera"]):
            skip("camera", "not on USB")
        else:
            actions.append(Node(package="v4l2_camera", executable="v4l2_camera_node", namespace="arm_camera",
                                output="screen", parameters=[{"video_device": dev["arm_camera"]["device"],
                                                              "image_size": [640, 480],
                                                              "camera_frame_id": "wrist_camera_link"}]))

    if "depth" in parts:
        if auto and not plugged_in(dev["depth_camera"]):
            skip("depth", "OAK-D not on USB")
        else:
            m = mounts.get("depth_camera", {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})
            actions += [
                IncludeLaunchDescription(PythonLaunchDescriptionSource(
                    os.path.join(get_package_share_directory("depthai_ros_driver"), "launch", "camera.launch.py")),
                    launch_arguments={"name": "oak", "parent_frame": "base_link",
                                      "cam_pos_x": str(m["xyz"][0]), "cam_pos_y": str(m["xyz"][1]),
                                      "cam_pos_z": str(m["xyz"][2]), "cam_roll": str(m["rpy"][0]),
                                      "cam_pitch": str(m["rpy"][1]), "cam_yaw": str(m["rpy"][2])}.items()),
            ]

    if "doctor" in parts:
        actions.append(Node(package="iro_ros", executable="doctor_node", output="screen"))

    if "rviz" in parts:
        if not os.environ.get("DISPLAY"):
            skip("rviz", "no screen (DISPLAY not set)")
        else:
            actions.append(Node(package="rviz2", executable="rviz2", output="screen",
                                arguments=["-d", str(share / "rviz" / "iro.rviz")]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("parts", default_value="all", description=f"'all' or a comma list of {ALL}"),
        OpaqueFunction(function=setup),
    ])
