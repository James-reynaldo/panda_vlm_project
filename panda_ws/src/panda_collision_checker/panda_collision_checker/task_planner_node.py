import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from .task_planner import TaskPlanner

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.task_planner = TaskPlanner(grasp_offset=0.10)
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'cube_pose',
            self.cube_pose_callback,
            10
        )

        self.grasp_pose_publisher = self.create_publisher(
            PoseStamped,
            'grasp_pose',
            10
        )
        self.get_logger().info("Task Planner Node initialized.")

    def cube_pose_callback(self, msg):
        # Update the task planner with the new cube pose
        cube_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        cube_ori = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        grasp_position, grasp_orientation = self.task_planner.generate_grasp_pose(cube_pos, cube_ori)
        grasp_msg = PoseStamped()
        grasp_msg.header = msg.header

        grasp_msg.pose.position.x = grasp_position[0]
        grasp_msg.pose.position.y = grasp_position[1]
        grasp_msg.pose.position.z = grasp_position[2]
        grasp_msg.pose.orientation.x = grasp_orientation[0]
        grasp_msg.pose.orientation.y = grasp_orientation[1]
        grasp_msg.pose.orientation.z = grasp_orientation[2]
        grasp_msg.pose.orientation.w = grasp_orientation[3]

        self.grasp_pose_publisher.publish(grasp_msg)

        self.get_logger().info(
            f"Frame ID: {msg.header.frame_id}, "
            f"Published grasp pose: Position {grasp_position}, "
            f"Orientation {grasp_orientation}"
        )

def main(args=None):

    rclpy.init(args=args)

    node = TaskPlannerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()