import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import cv2
import open3d as o3d

from message_filters import Subscriber, ApproximateTimeSynchronizer

class OneShotSaver(Node):
    def __init__(self):
        super().__init__('export_one_img_and_pointcloud')

        self.bridge = CvBridge()
        self.saved = False

        # Set up synchronized subscribers
        self.image_sub = Subscriber(self, Image, '/sim_cam_color_0/image_color')
        self.pc_sub = Subscriber(self, PointCloud2, '/sim_LiDAR_depth/points')

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.pc_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info("Waiting for synchronized image and point cloud...")

    def synced_callback(self, img_msg, pc_msg):
        if self.saved:
            return  # Only save once

        self.get_logger().info("Synchronized data received. Saving...")

        try:
            # Save image
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            cv2.imwrite('output_image.png', img)

            # Save point cloud
            points = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
            pc_np = np.asarray([[x, y, z] for x, y, z in points], dtype=np.float32)

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_np)

            # Save as .plys
            o3d.io.write_point_cloud("output_pointcloud.ply", pcd)

            self.get_logger().info("Saved output_image.png and output_pointcloud.npy")

            self.saved = True
            self.get_logger().info("Shutting down node.")
            self.destroy_node()
            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Failed to process message: {e}")

def main(args=None):
    rclpy.init(args=args)
    saver = OneShotSaver()
    rclpy.spin(saver)

if __name__ == '__main__':
    main()