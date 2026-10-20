"""Doctor as a ROS 2 node: runs `iro doctor` checks and publishes them on /diagnostics.

Risk checks (power, heat, servos) every `watch_period` seconds; the device and
backup checks every `full_period`. Shows up in rqt_robot_monitor and RViz.
Devices already opened by other nodes (arm, lidar) are reported as in use,
not probed, so the doctor never fights the drivers for a port.
"""

import json

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node

from iro import doctor

LEVEL = {doctor.OK: DiagnosticStatus.OK, doctor.INFO: DiagnosticStatus.OK, doctor.SKIP: DiagnosticStatus.OK,
         doctor.WARN: DiagnosticStatus.WARN, doctor.FAIL: DiagnosticStatus.ERROR}


class DoctorNode(Node):
    def __init__(self):
        super().__init__("doctor")
        self.declare_parameter("watch_period", 5.0)
        self.declare_parameter("full_period", 60.0)
        self.pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_timer(self.get_parameter("watch_period").value, lambda: self.check("watch"))
        self.create_timer(self.get_parameter("full_period").value, lambda: self.check("quick"))
        self.check("quick")

    def check(self, mode):
        results = doctor.run(mode=mode)
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        for r in results:
            values = [KeyValue(key="fix", value=r.fix)] if r.fix else []
            values += [KeyValue(key=k, value=json.dumps(v, default=str)[:200]) for k, v in r.data.items()]
            arr.status.append(DiagnosticStatus(level=LEVEL[r.status], name=f"doctor/{r.group}/{r.name}",
                                               message=r.detail, values=values))
            if r.status == doctor.FAIL:
                self.get_logger().error(f"{r.group}/{r.name}: {r.detail} -> {r.fix}", throttle_duration_sec=30.0)
        self.pub.publish(arr)


def main():
    rclpy.init()
    node = DoctorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
