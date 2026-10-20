"""Wrist camera node. The "openaicam" only offers MJPG, which v4l2_camera cannot
decode, so this reads it with OpenCV.

Publishes  <ns>/image_raw  sensor_msgs/Image (bgr8)
Params     device (from config/robot.yaml), width, height, fps, frame_id
"""

import array

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from iro import load_config


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera")
        dev = load_config()["devices"]["arm_camera"]["device"]
        self.declare_parameter("device", dev)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "wrist_camera_link")
        p = {n: self.get_parameter(n).value for n in ("device", "width", "height", "fps", "frame_id")}

        self.cap = cv2.VideoCapture(p["device"], cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, p["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p["height"])
        self.cap.set(cv2.CAP_PROP_FPS, p["fps"])
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {p['device']}")
        self.frame_id = p["frame_id"]
        self.pub = self.create_publisher(Image, "image_raw", qos_profile_sensor_data)
        self.create_timer(1.0 / p["fps"], self.grab)
        self.get_logger().info(f"camera {p['device']} {p['width']}x{p['height']} @ {p['fps']} fps")

    def grab(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("no frame from camera", throttle_duration_sec=5.0)
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        # array('B') takes rclpy's fast path: 0.2 ms vs 149 ms for plain bytes (checked per byte)
        msg.data = array.array("B", frame.tobytes())
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
