from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from xarm_planner.xarm_planning_client import XArmPlanningClient
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc_utils
from geometry_msgs.msg import PoseArray, TwistStamped
from std_srvs.srv import Trigger




class Touch():
    '''
    This class is responsible for practicing gentle touch based on tactile feedback
    '''
    def __init__(self) -> None:
        self.touch_node = Node("touch_node")


        ################################## Spinning Setup ################################
        self.callback_group = ReentrantCallbackGroup()
        self.executor = MultiThreadedExecutor(2)
        self.executor.add_node(self.touch_node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True, args=())
        self.executor_thread.start()
        self.touch_node.create_rate(1.0).sleep()

        ################################## Servoing Setup ################################
        self.planning_client = XArmPlanningClient(cartesian_planning=False,
                                            pipeline_id="ompl",
                                            planner_id="TRRT",)
        

        self.servo_msg = TwistStamped()
        self.servo_msg.twist.linear.z = 0.05
        # servoing by moving the end effector forward slowly
        self.servo_msg.header.frame_id = self.planning_client.moveit2.end_effector_name()
        
        self.stop_msg = TwistStamped()
        self.stop_msg.header.frame_id = self.planning_client.moveit2.end_effector_name()

        self.start_servoing = True
        ################################## Publisher Setup ################################
        self.servo_publisher = self.touch_node.create_publisher(
            msg_type=TwistStamped,
            topic='servo_node/delta_twist_cmds',
            qos_profile=10,
        )
        
        ################################## Subscriber Setup ################################
        tactile_feedback_sub = self.touch_node.create_subscription(
        msg_type=PointCloud2,
        topic='/gelsight_capture/patch',
        callback=self._tactile_callback,
        qos_profile=10,)

        ################################## Service Setup ################################
        # self.touch_starter_server = self.touch_node.create_service(
        #     srv_type=Trigger,
        #     srv_name="/fusion_sense/start_touch",
        #     callback=self._touch_starter_callback,
        # )
        
        ################################## Client Setup ################################
        # this is just to start the moveit servo feature
        self._moveit_servo_starter_client = self.touch_node.create_client(
            srv_type=Trigger,
            srv_name="/servo_node/start_servo",
        )
        self._moveit_servo_starter_client.wait_for_service()
        self._moveit_servo_starter_client.call_async(Trigger.Request())

    def start_touch(self) -> None:
        '''
        Start the touch practice
        '''
        self.start_servoing = True
        

    def _tactile_callback(self, msg: PointCloud2) -> None:
        '''
        Callback function for the tactile feedback subscriber
        '''

        # post process the tactile point cloud
        field_names=[field.name for field in msg.fields]
        cloud_data = list(pc_utils.read_points(msg, skip_nans=True, field_names = field_names))
        print(cloud_data)

        if self.start_servoing:
            self.servo_publisher.publish(self.servo_msg)
        else:
            self.servo_publisher.publish(self.stop_msg)

    def _touch_starter_callback(self, request, response):
        '''
        Callback function for the touch starter service
        '''
        self.start_servoing = True
        return response
    
def main():
    rclpy.init()
    touch = Touch()

    
    dummy_node = Node("dummy_node")
    patch_publisher = dummy_node.create_publisher(
        msg_type=PointCloud2, 
        topic='/gelsight_capture/patch', 
        qos_profile=10)
    
    patch_msg = PointCloud2()
    




if __name__ == '__main__':
    main()
