import rclpy
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2, Image
from image_geometry import PinholeCameraModel
from rclpy.duration import Duration

import ros2_numpy
import message_filters
from cv_bridge import CvBridge
import cv2  

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calib_node')

        self.cam_info_sub = self.create_subscription(
                CameraInfo, '/sim_cam_color_0/camera_info', self.k_callback, 10
            )
        
        self.cam_model = PinholeCameraModel()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        
        self.cv_bridge = CvBridge()

        img_sub = message_filters.Subscriber(
            self, Image, '/sim_cam_color_0/image_color'
        )
        pc_sub = message_filters.Subscriber(
            self, PointCloud2, '/sim_LiDAR_depth/points'
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, pc_sub], 10, 0.05, allow_headerless=True
        )
        self.sync.registerCallback(self.sync_callback)

        self.proj_pub = self.create_publisher(
            Image, '/proj_image', 10
        )

    def k_callback(self, msg: CameraInfo):

        self.cam_model.fromCameraInfo(msg)
        K = self.cam_model.intrinsicMatrix()
        self.get_logger().info(f"Loaded camera matrix: \n{K}")

        # Get the current time stamp
        stamp = msg.header.stamp
        if not hasattr(self, 'T_cam_lidar'):
            T = self.get_extrinsics(stamp)
            if T is None:
                return
            self.T_cam_lidar = T
            self.get_logger().info(f"Extrinsics: \n{T}")

            # Destroy subscription after first message
            self.destroy_subscription(self.cam_info_sub)
            self.get_logger().info("Camera info subscription destroyed.")

    def get_extrinsics(self, stamp):

        target_frame = 'BlueBoat/ZedCam1'
        source_frame = 'LiDAR'
        timeout = Duration(seconds=1.0)

        if self.tf_buffer.can_transform(target_frame, source_frame, stamp, timeout):
            tf_stamped = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout)
            translation = tf_stamped.transform.translation
            rotation = tf_stamped.transform.rotation
            self.get_logger().info(f"Translation: {translation}")
            self.get_logger().info(f"Rotation: {rotation}")
            # combine translation and rotation into a 4x4 matrix
            transform_matrix = self.combine_rot_trans(rotation, translation)
        else:
            self.get_logger().error("Transform not available")
            return None
        
        return transform_matrix
    
    def combine_rot_trans(self, rotation, translation):
        q = [rotation.x, rotation.y, rotation.z, rotation.w]
        R_mat = R.from_quat(q).as_matrix()      # 3×3
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R_mat
        T[:3, 3] = [translation.x,
                    translation.y,
                    translation.z]
        return T
    
    def sync_callback(self, img_msg: Image, pc_msg: PointCloud2):

        if not hasattr(self, 'T_cam_lidar'):
            self.get_logger().warn("Waiting for camera model and extrinsics to be set.")
            return
        
        xyz = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)
        xyz = xyz['xyz'].astype(np.float32)

        # Transform the point cloud to the camera frame
        ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
        xyz_h = np.hstack((xyz, ones))
        xyz_in_cam = (self.T_cam_lidar @ xyz_h.T).T[:, :3]

        # remove points behind the camera
        mask = xyz_in_cam[:, 2] > 0.0
        xyz_in_cam = xyz_in_cam[mask]

        # Project into pixel coords
        K = np.array(self.cam_model.intrinsicMatrix())
        proj = (K @ xyz_in_cam.T).T
        uv = proj[:, :2] / proj[:, 2:]

        # draw on image
        cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        H, W = cv_img.shape[:2]
        for u, v in uv:
            ui = int(round(u))
            vi = int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                cv2.circle(cv_img, (ui, vi), 3, (0, 255, 0), -1)

        # publish image
        out = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        out.header = img_msg.header

        self.proj_pub.publish(out)
        self.get_logger().info(f"Published projected image with {len(xyz_in_cam)} points.")
            

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()