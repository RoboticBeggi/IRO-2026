"""View the PUSAKA model in RViz.

Three ways to drive the joints (pick one with the `source` argument):

    ros2 launch pusaka_description display.launch.py                 # sliders
    ros2 launch pusaka_description display.launch.py source:=mirror  # follow the real arm
    ros2 launch pusaka_description display.launch.py source:=static  # frozen at zero

  source=slider  (default) - a GUI window with one slider per joint. The model
                 moves as you drag; the real arm does NOT.
  source=mirror  - reads the physical servos and moves the model to match, live.
                 Read-only on the arm. Needs `pusaka` on PYTHONPATH.
  source=static  - joints held at the URDF zero pose (arm horizontal). No GUI.

Other args:  rviz:=false  to skip RViz.

Only ONE joint-state source runs at a time - running two makes them fight over
/joint_states and nothing appears to move.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("pusaka_description")
    robot_desc = (Path(pkg) / "urdf" / "pusaka.urdf").read_text()

    source = LaunchConfiguration("source")
    is_slider = IfCondition(PythonExpression(["'", source, "' == 'slider'"]))
    is_mirror = IfCondition(PythonExpression(["'", source, "' == 'mirror'"]))
    is_static = IfCondition(PythonExpression(["'", source, "' == 'static'"]))

    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="slider",
                              choices=["slider", "mirror", "static"]),
        DeclareLaunchArgument("rviz", default_value="true"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen",
             parameters=[{"robot_description": robot_desc}]),

        # --- exactly one of these publishes /joint_states -----------------
        Node(package="joint_state_publisher_gui",
             executable="joint_state_publisher_gui",
             condition=is_slider, output="screen"),

        Node(package="joint_state_publisher", executable="joint_state_publisher",
             condition=is_static, output="screen"),

        Node(package="pusaka_bringup", executable="joint_state_pub",
             condition=is_mirror, output="screen",
             parameters=[{"rate_hz": 15.0}]),

        Node(package="rviz2", executable="rviz2", output="screen",
             condition=IfCondition(LaunchConfiguration("rviz")),
             arguments=["-d", str(Path(pkg) / "rviz" / "pusaka.rviz")]),
    ])

