# 3D Printing Instructions

## Object
- The `.stl` for printing the Stanford Bunny can be found at `/assets/3d_printing/object/stanford_bunny.stl`.
- The print file can be found at `/assets/3d_printing/object/stanford_bunny.form`
- The printing is done on a Formlabs Form 3 SLA 3D printer. The material used are: 
    - Transparent bunny: Clear Resin V4
    - Black Bunny: Black Resin V4
    - Reflective Bunny: Tough 2000 V1 + Generic reflective paint from Amazon
- Some tips for printing:
    1. Try to align the supports at the bottom of the bunny, which is invisible during experiment and won't be touched, because supports will leave some artifacts (bump) on the surface.
    2. Try not to use the cure station for the transparent bunny, because curing will cause the clear resin to become yellowish.
    3. For the same reason, we did not cure the black and reflective bunnies as we want to maintain maximum comparability between them. Instead, we use a fan to blow on the three bunnies for 24 hours.

## Camera and Tactile Sensor Mount
- The `.stl` files for the mount that connect the sensor/camera to the robot are located at:
    - `/assets/3d_printing/sensor_mount/xArm6_Interface_V2.STL`
    - `/assets/3d_printing/sensor_mount/xArm6_Sensor_Mount_V2.STL`
- The design source file from Solidworks is also in the same folder
- The interface is to connect to the robot EEF flange, while the mount will carry the camera and the sensor. The mount will be insert on top of the interface.
- The assembly file, which put the models of mount, interface, sensor and camera together, can be found in the `/assets/3d_printing/sensor_mount/assembly` folder.
    - This can be used for visualization in RViz, or for collision avoidance. For collision avoidance, I recommend downsample the assembly stl to reduce face number. There is a sample in the provided xArm ROS2 package
- Two M2 screws to connect the GelSight Mini tactile sensor. Two M3 screws to connect the Intel RealSense D405 camera.
- **Important:** The dimension of the mount is measured in Solidworks and put to the configuration file in the [RealSense interface package](https://github.com/ai4ce/realsense_ROS2_interface) and the [GelSight tactile sensor interface package](https://github.com/ai4ce/gelsight_ROS2_interface), so that we can accurately acquire the pose of camera and the sensor in the world coordinate by adding offsets to the EEF's coordinate.
    - Therefore, if you modify the parts' dimension, please update the configuration in the ROS2 packages accordingly. Specifically, the `realsense_ROS2_interface/realsense_capture/config/calibration.yaml` and `gelsight_ROS2_interface/gelsight_capture/config/gsmini.yaml`.

## Object Mounting Platform
- The `.stl` file for the mounting platform that connect the object to the tripod can be found at:
    - `/assets/3d_printing/tripod_mount/Tripod_Plate_180MM_V2.STL`
    - `/assets/3d_printing/tripod_mount/Tripod_Plate_180MM_V2_LID.STL`
- The `Tripod_Plate_180MM_V2.STL` can be inserted into the groove of the Arca-Swiss plate as we mentioned in the Appendix of the Arxiv paper. The `LID` then connect to the former so object can be fastened on this platform.