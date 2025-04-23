#!/usr/bin/env python3
"""
ROS 2 node that subscribes to a LiDAR PointCloud2, clusters it with DBSCAN,
and publishes both visualization markers and cluster centroids.
"""

import numpy as np
import rclpy
import ros2_numpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray

from rbi_perception_pkg.settings import PCL_CENTROID_TOPIC, PCL_CLUSTER_TOPIC, PCL_TOPIC


class LidarClusterNode(Node):
    def __init__(self):
        super().__init__("lidar_cluster_node")

        # Subscribers & publishers
        self.sub = self.create_subscription(
            PointCloud2, PCL_TOPIC, self.cloud_callback, qos_profile_sensor_data
        )
        self.marker_pub = self.create_publisher(MarkerArray, PCL_CLUSTER_TOPIC, 10)
        self.centroid_pub = self.create_publisher(PointStamped, PCL_CENTROID_TOPIC, 10)

        # DBSCAN parameters (tune to your LiDAR density & scene scale)
        self.eps = 0.5  # maximum distance (meters) between points in a cluster
        self.min_samples = 10  # minimum number of points to form a cluster

        self.get_logger().info(
            f"LidarClusterNode ready: subscribing to {PCL_TOPIC}, "
            f"publishing markers & centroids. Available at {PCL_CLUSTER_TOPIC} and {PCL_CENTROID_TOPIC}"
        )

    def cloud_callback(self, msg: PointCloud2):
        self.get_logger().info("call")
        # Convert PointCloud2 to an (N,3) NumPy array (x, y, z)
        pc: np.ndarray = ros2_numpy.point_cloud2.pointcloud2_to_xyz_array(
            msg, remove_nans=True
        )

        self.get_logger().info(f"Received PointCloud2 with shape {pc.shape}")

        # Skip if too few points
        if pc.shape[0] < self.min_samples:
            return

        # Run DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pc)
        labels = clustering.labels_  # -1 indicates noise

        marker_array = MarkerArray()

        for cluster_id in np.unique(labels):
            if cluster_id < 0:
                # ignore noise
                continue

            # Extract points belonging to this cluster
            pts = pc[labels == cluster_id]
            centroid = pts.mean(axis=0)  # (x, y, z)

            # Publish centroid as PointStamped
            ps = PointStamped()
            ps.header = msg.header
            ps.point.x, ps.point.y, ps.point.z = centroid
            self.centroid_pub.publish(ps)

            # Create a sphere marker at the centroid
            marker = Marker()
            marker.header = msg.header
            marker.ns = "lidar_clusters"
            marker.id = int(cluster_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5  # diameter in meters
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        # Publish all cluster markers at once
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down LidarClusterNode")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
