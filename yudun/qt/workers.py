import json
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from ..pipeline import Pipeline, read_jsonl
from ..simulation import generate


class ReplayWorker(QThread):
    ready = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, config, mode, scenario, path=None):
        super().__init__()
        self.config, self.mode, self.scenario, self.path = config, mode, scenario, path

    def run(self):
        try:
            inputs = list(read_jsonl(self.path)) if self.path else generate(self.config, self.scenario)
            pipeline, results = Pipeline(self.config, self.mode), []
            for frame in inputs:
                if self.isInterruptionRequested():
                    return
                results.append(pipeline.process(frame))
            self.ready.emit(inputs, results)
        except Exception as exc:
            self.failed.emit(str(exc))


class RosWorker(QThread):
    result = pyqtSignal(object)
    image = pyqtSignal(object)
    cloud = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config, mode, calibration=None, log_directory=None):
        super().__init__()
        self.config, self.mode, self.calibration = config, mode, calibration
        self.log_directory = Path(log_directory) if log_directory else None

    def run(self):
        nodes, executor, context = [], None, None
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.context import Context
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image, PointCloud2
            from sensor_msgs_py import point_cloud2
            from cv_bridge import CvBridge
            from yudun_ros.fusion import FusionNode
            from yudun_ros.sensors import SensorNode
            context = Context()
            rclpy.init(args=[], context=context)
            executor = SingleThreadedExecutor(context=context)
            fusion = FusionNode(self.config, self.mode, self.accept, context=context)
            nodes.append(fusion)
            raw = Node("yudun_qt_preview", context=context)
            nodes.append(raw)
            bridge = CvBridge()
            calibration = json.loads(Path(self.calibration).read_text(encoding="utf-8")) if self.calibration else {}
            camera = calibration.get("camera", {})
            lidar = calibration.get("lidar", {})
            self.subscriptions = [raw.create_subscription(Image, camera.get("topic", "/sensors/camera/image"),
                              lambda msg: self.image.emit(bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()), qos_profile_sensor_data),
                             raw.create_subscription(PointCloud2, lidar.get("topic", "/sensors/lidar/points"),
                              lambda msg: self.cloud.emit(point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)[::5].copy()), qos_profile_sensor_data)]
            if self.calibration:
                nodes.append(SensorNode(calibration_path=self.calibration, context=context))
            for node in nodes:
                executor.add_node(node)
            while not self.isInterruptionRequested() and context.ok():
                executor.spin_once(timeout_sec=.1)
        except Exception as exc:
            self.failed.emit(f"ROS 2 接入失败：{exc}")
        finally:
            if executor:
                executor.shutdown(timeout_sec=1)
            for node in reversed(nodes):
                node.destroy_node()
            if context and context.ok():
                context.shutdown()

    def accept(self, frame, result):
        if self.log_directory:
            self.log_directory.mkdir(parents=True, exist_ok=True)
            for filename, value in (("input.jsonl", frame), ("results.jsonl", result)):
                with (self.log_directory/filename).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        self.result.emit(result)
