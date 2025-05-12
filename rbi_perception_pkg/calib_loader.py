import rclpy
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel
from rclpy.duration import Duration

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calib_node')

        self.K = self.create_subscription(
                CameraInfo, '/sim_cam_color_0/camera_info', self.k_callback, 10
            )
        
        self.cam_model = PinholeCameraModel()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def k_callback(self, msg: CameraInfo):

        self.cam_model.fromCameraInfo(msg)
        K = self.cam_model.intrinsicMatrix()
        self.get_logger().info(f"Loaded camera matrix: \n{K}")

        # Get the current time stamp
        stamp = msg.header.stamp
        extrinsicts = self.get_extrinsics(stamp)
        self.get_logger().info(f"Extrinsics: \n{extrinsicts}")

        # Destroy subscription after first message
        self.destroy_subscription(self.K)
        self.get_logger().info("Camera info subscription destroyed.")

    def get_extrinsics(self, stamp):

        target_frame = 'BlueBoat/DcamF'
        source_frame = 'BlueBoat/ZedCam1'
        timeout = Duration(seconds=1.0)

        if self.tf_buffer.can_transform(target_frame, source_frame, stamp, timeout):
            tf_stamped = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp, timeout)
            translation = tf_stamped.transform.translation
            rotation = tf_stamped.transform.rotation
            self.get_logger().info(f"Translation: {translation}")
            self.get_logger().info(f"Rotation: {rotation}")
            # combine translation and rotation into a 4x4 matrix
            transform_matrix = self.combine_rot_trans(rotation, translation)
            self.get_logger().info(f"Transform matrix: {transform_matrix}")
        else:
            self.get_logger().error("Transform not available")
            return None
        
        return transform_matrix
    
# ros2 run tf2_ros static_transform_publisher --x 0.297 --y -0.425 --z 0.296 --frame-id BlueBoat/ZedCam2 --child-frame-id BlueBoat/DcamF
    
    def combine_rot_trans(self, rotation, translation):
        # Combine rotation and translation into a 4x4 matrix
        transform_matrix = [
            [1, 0, 0, translation.x],
            [0, 1, 0, translation.y],
            [0, 0, 1, translation.z],
            [0, 0, 0, 1]
        ]
        # Add rotation to the transform matrix
        transform_matrix[0][0] = 1 - 2 * (rotation.y * rotation.y + rotation.z * rotation.z)
        transform_matrix[0][1] = 2 * (rotation.x * rotation.y - rotation.z * rotation.w)
        transform_matrix[0][2] = 2 * (rotation.x * rotation.z + rotation.y * rotation.w)
        transform_matrix[1][0] = 2 * (rotation.x * rotation.y + rotation.z * rotation.w)
        transform_matrix[1][1] = 1 - 2 * (rotation.x * rotation.x + rotation.z * rotation.z)
        transform_matrix[1][2] = 2 * (rotation.y * rotation.z - rotation.x * rotation.w)
        transform_matrix[2][0] = 2 * (rotation.x * rotation.z - rotation.y * rotation.w)
        transform_matrix[2][1] = 2 * (rotation.y * rotation.z + rotation.x * rotation.w)
        transform_matrix[2][2] = 1 - 2 * (rotation.x * rotation.x + rotation.y * rotation.y)
        transform_matrix[3][3] = 1
        return transform_matrix


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