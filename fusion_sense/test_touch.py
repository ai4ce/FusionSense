from fusion_sense.touch import Touch
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc_utils
from std_msgs.msg import Header

import numpy as np

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]



def main():
    rclpy.init()

    def _tactile_callback_1():
        x = np.arange(320)
        y = np.arange(280)
        X, Y = np.meshgrid(x, y)
        points = np.zeros([320 * 280, 3])
        points[:, 0] = np.ndarray.flatten(X)  
        points[:, 1] = np.ndarray.flatten(Y)  
        points[:, 2] = np.zeros(320 * 280)
        header = Header()
        header.frame_id = 'gelsight_pointcloud'
        header.stamp = rclpy.get_clock().now().to_msg()
        patch_msg = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=points)
        patch_publisher.publish(patch_msg)
    
    def _tactile_callback_2():
        x = np.arange(320)
        y = np.arange(280)
        X, Y = np.meshgrid(x, y)
        points = np.zeros([320 * 280, 3])
        points[:, 0] = np.ndarray.flatten(X)  
        points[:, 1] = np.ndarray.flatten(Y)  
        points[:, 2] = np.zeros(320 * 280)
        points[0, 2] = 0.1
        points[0, 0] = 10
        header = Header()
        header.frame_id = 'gelsight_pointcloud'
        header.stamp = rclpy.get_clock().now().to_msg()
        patch_msg = pc_utils.create_cloud(header=header, fields=FIELDS_XYZ, points=points)
        patch_publisher.publish(patch_msg)



    touch = Touch()
    touch.start_touch()
    dummy_node = Node("dummy_node")
    patch_publisher = dummy_node.create_publisher(
        msg_type=PointCloud2, 
        topic='/gelsight_capture/patch', 
        qos_profile=10)
    
    dummy_node.create_timer(0.1, _tactile_callback_1)
    # stop the aforementioned timer after 10 seconds
    dummy_node.create_timer(10, lambda: dummy_node.destroy_timer(0))

    dummy_node.create_timer(10.1, _tactile_callback_2)

    rclpy.spin(dummy_node)

    




if __name__ == '__main__':
    main()
