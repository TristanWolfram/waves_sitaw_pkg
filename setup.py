from setuptools import find_packages, setup

package_name = "rbi_perception_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "rclpy",
        "sensor_msgs",
        "cv_bridge",
        "ultralytics",
        "opencv-python",
    ],
    zip_safe=True,
    maintainer="tristan",
    maintainer_email="tris.wolfram@googlemail.com",
    description="TODO: Package description",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_detector = rbi_perception_pkg.yolo_detection_node:main"
        ],
    },
)
