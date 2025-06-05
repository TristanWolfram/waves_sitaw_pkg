import rclpy
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
from geometry_msgs.msg import Point
from image_geometry import PinholeCameraModel
from ultralytics import YOLO

from rbi_perception_pkg.settings import MODEL_WEIGHTS

import ros2_numpy
import message_filters
from cv_bridge import CvBridge
import cv2  

from rbi_perception_pkg.Tracker import CombinedTracker

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calib_node')

        # Subscribtions
        self.cam_info_sub = self.create_subscription(
                CameraInfo, '/sim_cam_color_0/camera_info', self.k_callback, 10
            )
        img_sub = message_filters.Subscriber(
            self, Image, '/sim_cam_color_0/image_color'
        )
        pc_sub = message_filters.Subscriber(
            self, PointCloud2, '/sim_LiDAR_depth/points'
        )

        # Publisher
        self.proj_pub = self.create_publisher(
            Image, '/proj_image', 10
        )
        self.frustum_pub = self.create_publisher(
            MarkerArray, '/detection_frustums', 10
        )
        self.max_depth = 170.0

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, pc_sub], 10, 0.05, allow_headerless=True
        )
        self.sync.registerCallback(self.sync_callback)
        
        # YOLO
        self.yolo_model = YOLO(MODEL_WEIGHTS)
        self.get_logger().info(f"Loaded YOLO weights: {MODEL_WEIGHTS}")

        # Helper objects
        self.cam_model = PinholeCameraModel()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cv_bridge = CvBridge()

        # Tracking
        self.tracker = CombinedTracker(
            iou_weight=0.5,      # adjust 0–1 to favor IoU vs. appearance
            dist_thresh=0.7,     # maximum matching cost
            max_missed=5         # drop tracks after 5 missing frames
        )

        self.track_histories = {}
        self.max_history_length = 50


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
        
        cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        det_results = self.yolo_model(
            cv_img,
            verbose=False,
            show=False,
            )[0]
        
        xyz = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)
        xyz = xyz['xyz'].astype(np.float32)

        # PROJECTION
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

        tracks = self.tracker.update(det_results, cv_img)
        active_ids = { t["id"] for t in tracks }
        for tid in list(self.track_histories):
            if tid not in active_ids:
                del self.track_histories[tid]

        # VISUALIZATION
        # ---------------------------------------------------------------------------------------------------------
        k_inv = np.linalg.inv(K)
        T_lidar_cam = np.linalg.inv(self.T_cam_lidar)
        origin_lidar = (T_lidar_cam @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        frustum_markers = MarkerArray()

        # Crate frustum pyramid for each detection
        for idx, box in enumerate(det_results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pix = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
            )
            dirs = (k_inv @ np.c_[pix, np.ones(4)].T).T
            dirs /= dirs[:, 2:3]
            far_cam = dirs * self.max_depth
            far_lidar = (
                T_lidar_cam @ np.c_[far_cam, np.ones(4)].T
            ).T[:, :3]

            marker = Marker()
            marker.header = img_msg.header
            marker.header.frame_id = 'LiDAR'
            marker.ns = 'frustums'
            marker.id = int(idx)
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = Duration(seconds=0.5).to_msg()

            def add_line(p1, p2):
                pt1 = Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
                pt2 = Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))
                marker.points.append(pt1)
                marker.points.append(pt2)

            for p in far_lidar:
                add_line(origin_lidar, p)

            for i in range(4):
                p1 = far_lidar[i]
                p2 = far_lidar[(i + 1) % 4]
                add_line(p1, p2)

            frustum_markers.markers.append(marker)

        H, W = cv_img.shape[:2]

        for track in tracks:
            tid = track['id']
            x1, y1, x2, y2 = map(int, track['box'])

            hist = self.track_histories.setdefault(tid, [])
            if len(hist) > self.max_history_length:
                hist.pop(0)

            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(cv_img, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            inside_point = (
                (uv[:, 0] > x1) & (uv[:, 0] < x2) &
                (uv[:, 1] > y1) & (uv[:, 1] < y2)
            )
            uv_box = uv[inside_point]
            for u, v in uv_box:
                ui = int(round(u))
                vi = int(round(v))
                if 0 <= ui < W and 0 <= vi < H:
                    cv2.circle(cv_img, (ui, vi), 3, (0, 255, 0), -1)

        # publish image
        out = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        out.header = img_msg.header

        self.proj_pub.publish(out)
        self.frustum_pub.publish(frustum_markers)
        self.get_logger().info(
            f"Published projected image with {len(xyz_in_cam)} points and {len(det_results.boxes)} detections."
        )
        # ---------------------------------------------------------------------------------------------------------
            

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
