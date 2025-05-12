#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloRos2Node(Node):
    def __init__(self):
        super().__init__('yolo_ros2_node')
        self.declare_parameter('model_path', 'yolo12n.pt')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)

        mp     = self.get_parameter('model_path').get_parameter_value().string_value
        c_th   = self.get_parameter('conf_threshold').get_parameter_value().double_value
        i_th   = self.get_parameter('iou_threshold').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.model  = YOLO(mp)
        self.model.conf     = c_th
        self.model.iou      = i_th

        # Subscribe to your raw camera topic
        self.image_sub = self.create_subscription(
            Image, '/sim_cam_color_0/image_color', self.image_cb, 1)

        # Publish Detection2DArray
        self.det_pub   = self.create_publisher(
            Detection2DArray, '/perception/detections_str', 1)

        self.get_logger().info(f'YOLO node ready: model={mp}')

    def image_cb(self, img_msg: Image):
        # 1) stamp output with the image header
        det_arr = Detection2DArray()
        det_arr.header = img_msg.header

        # 2) convert to CV2 image
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')

        # 3) run inference
        results = self.model(cv_img)[0]

        # 4) pack results
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score         = float(box.conf[0])
            cls_id        = int(box.cls[0])

            bb = BoundingBox2D()
            bb.center.x = (x1 + x2) * 0.5
            bb.center.y = (y1 + y2) * 0.5
            bb.size_x   = x2 - x1
            bb.size_y   = y2 - y1

            det    = Detection2D()
            det.bbox = bb
            det.results.resize(1)
            det.results[0].id    = cls_id
            det.results[0].score = score

            det_arr.detections.append(det)

        # 5) publish
        self.det_pub.publish(det_arr)

def main(args=None):
    rclpy.init(args=args)
    node = YoloRos2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
