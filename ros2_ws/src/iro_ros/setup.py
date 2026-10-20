from setuptools import setup

setup(
    name="iro_ros",
    version="0.1.0",
    packages=["iro_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/iro_ros"]),
        ("share/iro_ros", ["package.xml"]),
    ],
    entry_points={"console_scripts": [
        "arm_node = iro_ros.arm_node:main",
        "doctor_node = iro_ros.doctor_node:main",
    ]},
)
