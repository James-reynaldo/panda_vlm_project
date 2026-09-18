import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from panda_interfaces.srv import PlanMotion
from .task_planner import TaskPlanner

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.task_planner = TaskPlanner(grasp_offset=0.10)
        self.latest_cube_pose = None
        self.planning_requested = False
        self.waiting_for_plan = False

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'cube_pose',
            self.cube_pose_callback,
            10
        )

        self.plan_motion_client = self.create_client(PlanMotion, '/plan_motion')
        self.plan_timer = self.create_timer(10.0, self._periodic_plan_request)
        self.get_logger().info("Task Planner Node initialized.")

    def cube_pose_callback(self, msg):
        # Store only the latest cube pose; do not trigger motion planning here.
        self.latest_cube_pose = msg
        self.get_logger().debug(
            f"Updated latest cube pose: frame_id={msg.header.frame_id}, "
            f"position=({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})"
        )

    def _periodic_plan_request(self):
        self.request_motion_plan()

    def request_motion_plan(self):
        if self.latest_cube_pose is None:
            self.get_logger().warn("No cube pose available to request a motion plan.")
            return

        if self.waiting_for_plan:
            self.get_logger().warn("A motion-plan request is already in progress; skipping this request.")
            return

        self.planning_requested = True
        self.waiting_for_plan = True

        cube_msg = self.latest_cube_pose
        cube_pos = np.array([
            cube_msg.pose.position.x,
            cube_msg.pose.position.y,
            cube_msg.pose.position.z,
        ])
        cube_ori = np.array([
            cube_msg.pose.orientation.x,
            cube_msg.pose.orientation.y,
            cube_msg.pose.orientation.z,
            cube_msg.pose.orientation.w,
        ])
        grasp_position, grasp_orientation = self.task_planner.generate_grasp_pose(cube_pos, cube_ori)

        request = PlanMotion.Request()
        request.grasp_pose.header = cube_msg.header
        request.grasp_pose.pose.position.x = grasp_position[0]
        request.grasp_pose.pose.position.y = grasp_position[1]
        request.grasp_pose.pose.position.z = grasp_position[2]
        request.grasp_pose.pose.orientation.x = grasp_orientation[0]
        request.grasp_pose.pose.orientation.y = grasp_orientation[1]
        request.grasp_pose.pose.orientation.z = grasp_orientation[2]
        request.grasp_pose.pose.orientation.w = grasp_orientation[3]

        if not self.plan_motion_client.wait_for_service(timeout_sec=5.0):
            self.waiting_for_plan = False
            self.planning_requested = False
            self.get_logger().error("/plan_motion service not available.")
            return

        future = self.plan_motion_client.call_async(request)
        future.add_done_callback(self._plan_response_callback)

        self.get_logger().info(
            f"Requested motion plan for cube pose: frame_id={cube_msg.header.frame_id}, "
            f"grasp_position={grasp_position}, grasp_orientation={grasp_orientation}"
        )

    def _plan_response_callback(self, future):
        self.waiting_for_plan = False

        try:
            response = future.result()
        except Exception as exc:
            self.planning_requested = False
            self.get_logger().error(f"/plan_motion service call failed: {exc}")
            return

        self.planning_requested = False
        self.get_logger().info(
            f"Plan response: success={response.success}, message={response.message}, "
            f"trajectory_points={len(response.trajectory.points)}"
        )

        if response.success:
            self.get_logger().info("Motion plan succeeded; can request a new plan when appropriate.")
        else:
            self.get_logger().warn("Motion plan failed; can request a new plan when appropriate.")

def main(args=None):

    rclpy.init(args=args)

    node = TaskPlannerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()