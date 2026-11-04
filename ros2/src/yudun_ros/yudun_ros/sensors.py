"""Raw ROS sensor inputs to calibrated map observations. Requires real calibration/model."""
import json
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yudun_interfaces.msg import DetectionArray, Detection3D
from yudun.adapters import OnnxDetector, camera_observation
from yudun.camera import load_camera


class SensorNode(Node):
    def __init__(self, calibration_path=None, context=None):
        super().__init__("yudun_sensors", context=context)
        self.declare_parameter("calibration", "")
        path = calibration_path or self.get_parameter("calibration").value
        if not path:
            raise ValueError("calibration JSON is required; do not use example calibration on real hardware")
        self.calibration = {"camera": load_camera(path)}
        self.bridge = CvBridge()
        self.camera_pub = self.create_publisher(DetectionArray, "/perception/visual_detections", qos_profile_sensor_data)
        camera = self.calibration.get("camera")
        self.detector = OnnxDetector(camera["model"], camera["class_names"], camera.get("input_size", 640),
                                     **camera.get("detector_options", {})) if camera else None
        self.subs = []
        if camera:
            self.subs.append(self.create_subscription(Image, camera.get("topic", "/sensors/camera/image"), self.image, qos_profile_sensor_data))

    def publish(self, publisher, header, sensor, observations):
        msg = DetectionArray()
        msg.header.stamp, msg.header.frame_id, msg.sensor = header.stamp, "map", sensor
        for obs in observations:
            detection = Detection3D()
            detection.header = msg.header
            detection.observation_id, detection.category = obs.observation_id, obs.category
            detection.confidence, detection.quality = float(obs.confidence), float(obs.quality)
            detection.x, detection.y = map(float, obs.position)
            detection.covariance = np.asarray(obs.covariance).reshape(-1).tolist()
            detection.radius = float(obs.radius)
            msg.detections.append(detection)
        publisher.publish(msg)

    def image(self, msg):
        try:
            c = self.calibration["camera"]
            if msg.header.frame_id != c["frame_id"]:
                raise ValueError("camera frame mismatch")
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if image.shape[:2] != (c["image_height"], c["image_width"]):
                raise ValueError("camera image resolution differs from calibration")
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
            observations = [camera_observation(d["bbox"], d["category"], d["confidence"], stamp,
                            c["intrinsics"], c["world_from_camera"], c.get("measurement_covariance"), c.get("distortion"),
                            f"{stamp:.9f}:{i}") for i, d in enumerate(self.detector.detect(image))]
            self.publish(self.camera_pub, msg.header, "camera", observations)
        except (ValueError, RuntimeError) as exc:
            self.get_logger().error(f"Camera adapter failed: {exc}")

def main():
    rclpy.init()
    node = SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
