# settings.py
"""
Configuration parameters for YOLO detector node.
Add more settings here as needed.
"""

# ROS topic names
IMAGE_TOPIC = "/sim_cam_color_0/image_color"
DETECTION_TOPIC = "/perception/detections"

# YOLO model weights
MODEL_WEIGHTS = "yolo12n.pt"

# Other potential future settings
# FRAME_RATE = 10  # in Hz
# CONFIDENCE_THRESHOLD = 0.5


# Pointcloud settings

PCL_TOPIC = "/sim_LIDAR_depth/points"
PCL_CLUSTER_TOPIC = "/sim_points/clusters"
PCL_CENTROID_TOPIC = "/sim_points/centroids"
