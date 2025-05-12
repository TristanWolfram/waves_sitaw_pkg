#!/usr/bin/env python3

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import cv2

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
        annotated = cv_image.copy()
        if results.boxes:
            # Draw bounding boxes on the image
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
