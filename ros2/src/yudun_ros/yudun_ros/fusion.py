import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from yudun_interfaces.msg import DetectionArray, FusedTrackArray, FusedTrack, RiskEventArray, RiskEvent
from yudun.schema import Observation
from yudun.adapters import observations_frame
from yudun.pipeline import Pipeline, load_config


def seconds(stamp):
    return stamp.sec + stamp.nanosec/1e9


class FusionNode(Node):
    def __init__(self, config=None, mode=None, on_result=None, context=None):
        super().__init__("yudun_fusion", context=context)
        for key, value in (("config", ""), ("mode", "monocular")):
            self.declare_parameter(key, value)
        self.pipeline = Pipeline(config or load_config(self.get_parameter("config").value or None), mode or self.get_parameter("mode").value)
        self.on_result = on_result
        self.pending, self.heartbeats, self.last_sensor_stamp = {}, set(), {}
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.track_pub = self.create_publisher(FusedTrackArray, "/tracking/fused_tracks", reliable)
        self.risk_pub = self.create_publisher(RiskEventArray, "/safety/risk_events", reliable)
        self.subscriptions_list = [self.create_subscription(DetectionArray, topic, self.receive, qos_profile_sensor_data)
                                   for topic in ("/perception/visual_detections",)]
        self.create_timer(0.1, self.tick)

    def receive(self, msg):
        try:
            if msg.schema_version != 1 or msg.header.frame_id != "map" or msg.sensor != "camera":
                raise ValueError("invalid array schema/frame/sensor")
            now = self.get_clock().now().nanoseconds/1e9
            stamp = seconds(msg.header.stamp)
            if not 0 <= now-stamp <= self.pipeline.config["tracking"]["max_observation_age"]:
                raise ValueError("stale or future sensor heartbeat")
            if stamp <= self.last_sensor_stamp.get(msg.sensor, -1):
                raise ValueError("duplicate or out-of-order sensor acquisition")
            batch = []
            for detection in msg.detections:
                if detection.schema_version != 1 or detection.header.frame_id != "map":
                    raise ValueError("invalid detection version/frame")
                if abs(seconds(detection.header.stamp)-stamp) > 1e-6:
                    raise ValueError("array and detection acquisition stamps must agree")
                batch.append(Observation(seconds(detection.header.stamp), msg.sensor, [detection.x, detection.y],
                                         detection.category, detection.confidence,
                                         np.asarray(detection.covariance).reshape(2, 2).tolist(), detection.quality,
                                         observation_id=detection.observation_id, radius=detection.radius))
            # Keep only the newest complete acquisition per sensor per fusion tick.
            # Treating several camera frames as simultaneous births duplicates tracks.
            self.pending[msg.sensor] = batch
            self.last_sensor_stamp[msg.sensor] = stamp
            self.heartbeats.add(msg.sensor)
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"Rejected observations: {exc}")

    def tick(self):
        now = self.get_clock().now()
        timestamp = now.nanoseconds/1e9
        if self.pipeline.tracker.timestamp is not None and timestamp <= self.pipeline.tracker.timestamp:
            # Paused ROS simulation clock: do not age. Backward seeks require node restart.
            return
        frame = observations_frame(timestamp, [o for batch in self.pending.values() for o in batch], sorted(self.heartbeats))
        self.pending, self.heartbeats = {}, set()
        try:
            result = self.pipeline.process(frame)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        tracks = FusedTrackArray()
        tracks.header.stamp, tracks.header.frame_id = now.to_msg(), "map"
        tracks.system_status = result["system_status"]
        tracks.camera_status, tracks.lidar_status = result["health"].get("camera", "NOT_CONFIGURED"), result["health"].get("lidar", "NOT_CONFIGURED")
        for t in result["tracks"]:
            item = FusedTrack()
            item.header = tracks.header
            item.track_id, item.category, item.status = t["id"], t["category"], t["status"]
            item.state, item.covariance = t["state"], np.asarray(t["covariance"]).reshape(-1).tolist()
            item.last_seen, item.radius, item.sources = t["last_seen"], t["radius"], t["sources"]
            tracks.tracks.append(item)
        self.track_pub.publish(tracks)
        events = RiskEventArray()
        events.header = tracks.header
        for e in result["events"]:
            item = RiskEvent()
            item.header = events.header
            item.event_id, item.track_pair = e["event_id"], e["pair"]
            item.level, item.probability = e["level"], e.get("probability", 0.)
            item.tca, item.minimum_distance = e.get("tca", 0.), e.get("minimum_distance", 0.)
            item.transition, item.reason, item.predicted = e["transition"], e["reason"], e.get("predicted", False)
            events.events.append(item)
        self.risk_pub.publish(events)
        if self.on_result:
            self.on_result(frame, result)


def main():
    rclpy.init()
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
