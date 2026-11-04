from glob import glob

from setuptools import setup

setup(
    name="iro_bringup",
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/iro_bringup"]),
        ("share/iro_bringup", ["package.xml"]),
        ("share/iro_bringup/launch", glob("launch/*.py")),
        ("share/iro_bringup/rviz", glob("rviz/*.rviz")),
        ("share/iro_bringup/config", glob("config/*")),
        ("share/iro_bringup/urdf/so101", glob("urdf/so101/*.urdf")),
        ("share/iro_bringup/urdf/so101/assets", glob("urdf/so101/assets/*")),
    ],
)
