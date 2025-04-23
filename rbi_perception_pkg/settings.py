# settings.py
"""
Configuration parameters for YOLO detector node.
Add more settings here as needed.
"""

# ROS topic names
IMAGE_TOPIC = "/sim_cam_color_0/image_color"
DETECTION_TOPIC = "/sim_inmgs/detections"

# YOLO model weights
MODEL_WEIGHTS = "yolo11n.pt"

# Other potential future settings
# FRAME_RATE = 10  # in Hz
# CONFIDENCE_THRESHOLD = 0.5
