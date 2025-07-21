"""Visualization utilities for calibration node."""

from typing import List, Dict

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray


class CalibrationVisualizer:
    """Handle visualization and publishing of calibration results."""

    def __init__(self, node: Node, max_depth: float = 170.0) -> None:
        self._node = node
        self.max_depth = max_depth
        self.proj_pub = node.create_publisher(Image, '/proj_image', 10)
        self.frustum_pub = node.create_publisher(MarkerArray, '/detection_frustums', 10)
        self.centroid_pub = node.create_publisher(MarkerArray, '/cluster_centroids', 10)
        self.cv_bridge = CvBridge()
        self.track_histories: Dict[int, List] = {}
        self.max_history_length = 50

    def publish(self,
                img_msg: Image,
                cv_img: np.ndarray,
                det_results,
                track_infos: List[Dict],
                K: np.ndarray,
                T_lidar_cam: np.ndarray,
                num_points: int) -> None:
        """Publish visualization markers and annotated image."""
        k_inv = np.linalg.inv(K)
        origin_lidar = (T_lidar_cam @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        frustum_markers = MarkerArray()
        centroid_markers = MarkerArray()

        for idx, box in enumerate(det_results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pix = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
            dirs = (k_inv @ np.c_[pix, np.ones(4)].T).T
            dirs /= dirs[:, 2:3]
            far_cam = dirs * self.max_depth
            far_lidar = (T_lidar_cam @ np.c_[far_cam, np.ones(4)].T).T[:, :3]

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
        for info in track_infos:
            tid = info['tid']
            x1, y1, x2, y2 = info['bbox']

            hist = self.track_histories.setdefault(tid, [])
            if len(hist) > self.max_history_length:
                hist.pop(0)

            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(cv_img, f'ID: {tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            uv_box = info['uv_box']
            centroid_lidar = info['centroid']

            centroid_marker = Marker()
            centroid_marker.header = img_msg.header
            centroid_marker.header.frame_id = 'LiDAR'
            centroid_marker.ns = 'centroids'
            centroid_marker.id = int(tid)
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.scale.x = 1.0
            centroid_marker.scale.y = 1.0
            centroid_marker.scale.z = 1.0
            centroid_marker.color.r = 0.0
            centroid_marker.color.g = 1.0
            centroid_marker.color.b = 0.0
            centroid_marker.color.a = 1.0
            centroid_marker.pose.orientation.w = 1.0
            centroid_marker.pose.position.x = float(centroid_lidar[0])
            centroid_marker.pose.position.y = float(centroid_lidar[1])
            centroid_marker.pose.position.z = float(centroid_lidar[2])
            centroid_marker.lifetime = Duration(seconds=0.5).to_msg()
            centroid_markers.markers.append(centroid_marker)

            for u, v in uv_box:
                ui = int(round(u))
                vi = int(round(v))
                if 0 <= ui < W and 0 <= vi < H:
                    cv2.circle(cv_img, (ui, vi), 3, (0, 255, 0), -1)

        out = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        out.header = img_msg.header
        self.proj_pub.publish(out)
        self.frustum_pub.publish(frustum_markers)
        self.centroid_pub.publish(centroid_markers)

        self._node.get_logger().info(
            f'Published projected image with {num_points} points and '
            f'{len(det_results.boxes)} detections.'
        )
