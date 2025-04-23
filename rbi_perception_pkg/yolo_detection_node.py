#!/usr/bin/env python3
"""
Example ROS2 node that subscribes to a camera image topic, runs a YOLOv8 inference on each frame,
and publishes an annotated image showing detected objects.

Dependencies:
  - ROS2 (e.g. Humble or later)
  - rclpy
  - sensor_msgs
  - cv_bridge
  - ultralytics (pip install ultralytics)
  - OpenCV (pip install opencv-python)

Usage:
  1. Create a ROS2 Python package:
       ros2 pkg create --build-type ament_python yolo_detector
  2. Add dependencies to package.xml:
       <depend>rclpy</depend>
       <depend>sensor_msgs</depend>
       <depend>cv_bridge</depend>
  3. Copy this script into yolo_detector/yolo_detector/detect_node.py
  4. In setup.py, ensure entry point:
       entry_points={
         'console_scripts': [
           'detect_node = yolo_detector.detect_node:main',
         ],
       },
  5. Install requirements: pip install ultralytics opencv-python
  6. Build and source your workspace:
       colcon build --packages-select yolo_detector
       source install/setup.bash
  7. Run:
       ros2 run yolo_detector detect_node

This node listens on '/camera/image_raw' and publishes on '/camera/image_detections'.
"""
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

# Import from within the package
from rbi_perception_pkg.settings import DETECTION_TOPIC, IMAGE_TOPIC, MODEL_WEIGHTS


class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")

        # Load pretrained YOLO model
        self.model = YOLO(MODEL_WEIGHTS)
        self.bridge = CvBridge()

        # Subscriber and publisher use names from settings
        self.subscription = self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, 10
        )
        self.publisher = self.create_publisher(Image, DETECTION_TOPIC, 10)

        self.get_logger().info(
            f"Subscribed: {IMAGE_TOPIC} | Publishing: {DETECTION_TOPIC}"
        )

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Conversion failed: {e}")
            return

        results = self.model(cv_image)[0]
        annotated = results.plot()

        out_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        out_msg.header = msg.header
        self.publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
