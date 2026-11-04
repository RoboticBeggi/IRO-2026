"""RViz arm that follows the real arm, live (`iro mirror`). Read-only on the arm.

Rebuilt from the old PUSAKA `display.launch.py source:=mirror`.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    robot = os.path.join(get_package_share_directory("iro_bringup"), "launch", "robot.launch.py")
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(robot), launch_arguments={"parts": "arm,rviz"}.items()),
    ])
