# Robotics Software Installation

## ROS 2
1. Install ROS2 Humble according to the [official instruction](https://docs.ros.org/en/humble/Installation.html). 
    - Note that while this project is developed under Humble, we do not explicitly use any Humble-specific feature, so other distro should work in principle.

## RealSense D405 Depth Camera
2. Install the [RealSense depth camera interface package](https://github.com/ai4ce/realsense_ROS2_interface). This is a custom ROS2 package built upon Intel's official ROS2 wrapper.

## GelSight Mini Tactile Sensor
3. Install the [GelSight tactile sensor interface package](https://github.com/ai4ce/gelsight_ROS2_interface). 
    - **Unfortunately**, this package is built upon a codebase that has not been open-source. We will update the link within a few weeks when the dependency is fully open. Stay tuned!
    - Note that this is not the same as the official GelSight implementation. We made some tweaks and (hopefully) improvements, so you are strongly encouraged to use our own package.

**Note:** See [3D Printing Instructions](3d_printing.md) for some caveats on the dimension of the camera/sensor mount and the world-frame coordinates of the camera and the tactile sensor.

## Robot Servoing and Teleoperation.
4. Install the [Ufactory xArm6 servoing and teleoperation package](https://github.com/ai4ce/xarm_ros2). This is a custom ROS2 package built upon the official UFactory ROS2 packages.
    - We also have a [UR10e equivalent](https://github.com/ai4ce/ur_ros2) available. 
    - If you are using a different robot, while we may not have a ROS2 package readily available, as long as your robot works with MoveIt 2, you should be able to adapt my code fairly easily.