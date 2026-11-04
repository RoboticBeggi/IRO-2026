from setuptools import setup

setup(
    name="iro_ros",
    version="0.1.0",
    packages=["iro_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/iro_ros"]),
        ("share/iro_ros", ["package.xml"]),
        # Plain scripts, not console_scripts: those get a hard-coded /usr/bin/python3,
        # which cannot import LeRobot from our venv
        ("lib/iro_ros", ["scripts/arm_node", "scripts/camera_node", "scripts/doctor_node"]),
    ],
)
