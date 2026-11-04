"""SO-ARM101 ROS 2 node: owns the servo bus.

Publishes
  /joint_states   sensor_msgs/JointState  real arm angles (radians), so RViz mirrors the arm
  /diagnostics    diagnostic_msgs/DiagnosticArray  servo volts, temps, errors (1 Hz)
Services
  /arm/torque     std_srvs/SetBool   true = hold position, false = limp (teach by hand)

Read-only unless /arm/torque is called. Replaces the old PUSAKA joint_state_pub.
"""

import math

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from iro import load_config
from iro.hw.servos import open_bus, read_health

TICKS = 4096  # STS3215 counts per turn; LeRobot calibration puts mid-range at 2048 = URDF zero


class ArmNode(Node):
    def __init__(self):
        super().__init__("arm")
        cfg = load_config()
        arm = cfg["devices"]["arm"]
        self.lim = cfg["limits"]["servos"]
        self.signs = arm.get("joint_signs", {})
        self.offsets = {j: math.radians(v) for j, v in arm.get("joint_offsets_deg", {}).items()}
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("port", arm["port"])

        self._bus_cm = open_bus(self.get_parameter("port").value, arm["servo_ids"])
        self.bus = self._bus_cm.__enter__()
        self.names = list(self.bus.motors)
        # Live-adjustable zero offsets (degrees, in self.names order), used by `iro arm align`
        self.declare_parameter("joint_offsets_deg", [math.degrees(self.offsets.get(n, 0.0)) for n in self.names])
        self.add_on_set_parameters_callback(self.on_params)

        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_service(SetBool, "/arm/torque", self.on_torque)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.publish_joints)
        self.create_timer(1.0, self.publish_health)
        self.get_logger().info(f"arm ready on {self.get_parameter('port').value}, {len(self.names)} servos")

    def on_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "joint_offsets_deg":
                if len(p.value) != len(self.names):
                    return SetParametersResult(successful=False, reason=f"need {len(self.names)} values")
                self.offsets = {n: math.radians(v) for n, v in zip(self.names, p.value)}
        return SetParametersResult(successful=True)

    def publish_joints(self):
        try:
            raw = self.bus.sync_read("Present_Position", normalize=False)
        except Exception as e:  # noqa: BLE001 - a dropped packet must not kill the node
            self.get_logger().warn(f"position read failed: {e}", throttle_duration_sec=5.0)
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = [self.signs.get(n, 1) * (raw[n] - TICKS / 2) * 2 * math.pi / TICKS + self.offsets.get(n, 0.0)
                        for n in self.names]
        self.js_pub.publish(msg)

    def publish_health(self):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        try:
            health = read_health(self.bus)
        except Exception as e:  # noqa: BLE001
            arr.status.append(DiagnosticStatus(level=DiagnosticStatus.ERROR, name="arm/servos",
                                               message=f"servo bus read failed: {e}"))
            self.diag_pub.publish(arr)
            return
        for name, h in health.items():
            level, msg = DiagnosticStatus.OK, f"{h['temp_c']} C, {h['volts']:.1f} V"
            if h["errors"]:
                level, msg = DiagnosticStatus.ERROR, "servo reports: " + ", ".join(h["errors"])
            elif h["temp_c"] >= self.lim["temp_fail_c"] or h["volts"] < self.lim["volt_fail"]:
                level = DiagnosticStatus.ERROR
            elif h["temp_c"] >= self.lim["temp_warn_c"] or h["volts"] < self.lim["volt_warn"] \
                    or h["load_pct"] >= self.lim["load_warn_pct"]:
                level = DiagnosticStatus.WARN
            arr.status.append(DiagnosticStatus(
                level=level, name=f"arm/{name}", message=msg, hardware_id=f"sts3215-{self.bus.motors[name].id}",
                values=[KeyValue(key=k, value=str(v)) for k, v in h.items()]))
        self.diag_pub.publish(arr)

    def on_torque(self, req, res):
        try:
            (self.bus.enable_torque if req.data else self.bus.disable_torque)()
            res.success, res.message = True, "torque on" if req.data else "torque off (arm is limp)"
        except Exception as e:  # noqa: BLE001
            res.success, res.message = False, str(e)
        return res

    def destroy_node(self):
        self._bus_cm.__exit__(None, None, None)  # leaves torque as it is
        super().destroy_node()


def main():
    rclpy.init()
    node = ArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
