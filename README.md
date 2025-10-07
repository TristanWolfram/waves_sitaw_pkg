# WAVES - Waterborn Agents in Virtual Environment Simulations
## Perception Package

## Overview
`rbi_perception_pkg` provides ROS 2 nodes that subscribe to synchronized RGB images and LiDAR point clouds. Furthermore a pipline is used to perform object detection within images and sensor fusion to find the detected objects position. There are also a lot of testing scripts mainly for visualization and tryouts of different methods.

## Package layout
- `rbi_perception_pkg/`: Python package containing ROS 2 nodes and helpers.
  - `calib_loader.py`: **Main Pipeline** Calibration and visualisation node that aligns LiDAR and camera data, performs YOLO detections, and maintains multi-object tracks.
  - `yolo_detection_node.py`(*testing*): YOLO inference node that annotates incoming images.
  - `export_one_img_and_pointcloud.py`(*testing*): Utility node that saves one synchronized frame of image and point cloud data.
  - Additional helpers (`settings.py`, `Helpermethods.py`, `Tracker.py`, etc.) that provide configuration, tracking, and visualisation utilities.
- `src/base_subscriber.cpp`: Example C++ node that prints diagnostics when synchronized point cloud and image messages are received. ---> (*testing*) In the start I tried to realiese the package with C++. Becasue of YOLO I decided for Python in the end.

## Dependencies
### ROS 2 packages
- `rclpy`, `rclcpp`
- `sensor_msgs`, `visualization_msgs`, `geometry_msgs`
- `tf2_ros`
- `image_geometry`
- `message_filters`

### Python packages
Installed automatically via `setup.py` when you build the package:
- `ultralytics` (YOLO models)
- `opencv-python`
- `cv_bridge`
- `ros2-numpy`
- `scikit-learn`
- `open3d`
- Standard build tools: `setuptools`

## Building
1. Place the package inside a ROS 2 workspace (e.g., `~/ros2_ws/src/rbi_perception_pkg`).
2. Source your ROS 2 installation: `source /opt/ros/<distro>/setup.bash`.
3. Build with colcon:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select rbi_perception_pkg
   ```
4. Source the workspace overlay: `source install/setup.bash`.

## Configuration
Key topics and model parameters are defined in `rbi_perception_pkg/settings.py`:
- `IMAGE_TOPIC`: incoming RGB image topic (default `/sim_cam_color_0/image_color`).
- `DETECTION_TOPIC`: output topic for annotated detections (default `/perception/detections`).
- `MODEL_WEIGHTS`: YOLO weights file to load (`yolo11n.pt`, `yolo12n.pt`, etc.).
- Point cloud topics used by other nodes.

## Commands and usage
All commands assume your ROS 2 workspace has been built and sourced.

### Calibration loader and visualiser
Synchronizes RGB images and LiDAR point clouds, retrieves TF transforms, projects detections into 3D, and publishes visualizations.
```bash
ros2 run rbi_perception_pkg calib_loader
```
- Subscribes to: `/sim_cam_color_0/camera_info`, `/sim_cam_color_0/image_color`, `/sim_LiDAR_depth/points`.
- Requires valid TF frames (`BlueBoat/ZedCam1` to `LiDAR`) to compute extrinsics.
- Uses YOLO weights defined in `MODEL_WEIGHTS` and a combined IoU/appearance tracker to maintain object IDs.
- Outputs visualization messages via `CalibrationVisualizer` (see `calib_visualizer.py`).

### YOLO detector
Annotates incoming images with YOLO bounding boxes and republishes them.
```bash
ros2 run rbi_perception_pkg yolo_detector
```
- Subscribes to: `IMAGE_TOPIC` (default `/sim_cam_color_0/image_color`).
- Publishes to: `DETECTION_TOPIC` (default `/perception/detections`).
- Parameters: adjust weights or topics in `settings.py`.

### One-shot image and point cloud export
Saves a single synchronized image and LiDAR point cloud to disk and shuts down.
```bash
ros2 run rbi_perception_pkg export_one_img_and_pointcloud
```
- Subscribes to: `/sim_cam_color_0/image_color`, `/sim_LiDAR_depth/points`.
- Writes `output_image.png` and `output_pointcloud.ply` in the current working directory.
- Waits for the first synchronized pair of messages before saving.

## License
This project is released under the MIT License (see `LICENSE`).
