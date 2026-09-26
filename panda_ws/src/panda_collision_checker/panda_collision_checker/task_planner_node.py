import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from panda_interfaces.srv import PlanMotion
from .task_planner import TaskPlanner

class TaskPlannerNode(Node):
    # State machine states
    STATE_IDLE = 0
    STATE_WAITING_FOR_SETUP = 1
    STATE_WAITING_FOR_GRASP = 2
    STATE_WAITING_FOR_GRIPPER_CLOSE = 3
    STATE_WAITING_FOR_PRE_PLACE = 4
    STATE_WAITING_FOR_PLACE = 5
    STATE_DONE = 6

    def __init__(self):
        super().__init__('task_planner_node')
        # Setup planner: offset=0.13 for approach position
        self.setup_planner = TaskPlanner(grasp_offset=-0.13)
        # Grasp planner: offset=0 for actual grasp position
        self.grasp_planner = TaskPlanner(grasp_offset=-0.07)
        # Pre-place planner: approach the drawer with a 13 cm offset.
        self.pre_place_planner = TaskPlanner(grasp_offset=-0.13)
        self.latest_cube_pose = None
        self.current_state = self.STATE_IDLE
        self.setup_pose = None
        self.grasp_pose = None
        # Target inside the lower drawer, expressed in the cube_pose/world frame.
        self.lower_drawer_position = np.array([0.0, 0.1, 0.93])
        self.pre_place_pose = None
        self.place_pose = None
        self.gripper_close_timer = None


        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'cube_pose',
            self.cube_pose_callback,
            10
        )

        self.plan_motion_client = self.create_client(PlanMotion, '/plan_motion')
        
        # Subscribe to motion completion signal
        self.motion_completed_subscription = self.create_subscription(
            Bool,
            '/motion_completed',
            self.motion_completed_callback,
            10
        )

        self.gripper_command_publisher = self.create_publisher(Bool, '/gripper_command', 10)
        # Initial state transition: start with requesting setup pose
        self.setup_timer = self.create_timer(2.0, self._request_setup_motion)
        self.get_logger().info("Task Planner Node initialized. Waiting for cube pose to request setup motion.")

    def cube_pose_callback(self, msg):
        # Store the latest cube pose for use when generating motion plans
        self.latest_cube_pose = msg
        self.get_logger().debug(
            f"Updated latest cube pose: frame_id={msg.header.frame_id}, "
            f"position=({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})"
        )

    def _request_setup_motion(self):
        """Request motion to setup/pre-grasp pose."""
        if self.latest_cube_pose is None:
            self.get_logger().warn("No cube pose available yet; waiting for cube_pose message.")
            return

        if self.current_state != self.STATE_IDLE:
            # Timer is no longer needed; cancel it
            return

        self.current_state = self.STATE_WAITING_FOR_SETUP
        
        cube_msg = self.latest_cube_pose
        cube_pos = np.array([
            cube_msg.pose.position.x,
            cube_msg.pose.position.y,
            cube_msg.pose.position.z,
        ])
        cube_ori = np.array([ # In world frame, expressed as quaternion [x, y, z, w]
            cube_msg.pose.orientation.x,
            cube_msg.pose.orientation.y,
            cube_msg.pose.orientation.z,
            cube_msg.pose.orientation.w,
        ])

        # Generate setup pose using approach offset (0.13)
        setup_position, setup_orientation = self.setup_planner.generate_grasp_pose(cube_pos,cube_ori)
        self.setup_pose = (setup_position, setup_orientation)

        self._send_plan_request(self.setup_pose, "SETUP")

    def _request_grasp_motion(self):
        """Request motion to grasp pose."""
        if self.latest_cube_pose is None:
            self.get_logger().warn("No cube pose available; cannot request grasp motion.")
            self.current_state = self.STATE_IDLE
            return

        self.current_state = self.STATE_WAITING_FOR_GRASP

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

        # Generate grasp pose using zero offset (actual grasp position)
        grasp_position, grasp_orientation = self.grasp_planner.generate_grasp_pose(cube_pos, cube_ori)
        self.grasp_pose = (grasp_position, grasp_orientation)

        self._send_plan_request(self.grasp_pose, "GRASP")

    def _request_pre_place_motion(self):
        """Request motion to the approach pose above the lower drawer."""
        if self.grasp_pose is None:
            self.get_logger().error("No grasp pose is available; cannot request pre-place motion.")
            self.current_state = self.STATE_IDLE
            return

        self.current_state = self.STATE_WAITING_FOR_PRE_PLACE

        self.pre_place_pose = self.pre_place_planner.generate_approach_pose(
            self.lower_drawer_position,
            self.grasp_pose[1],
        )
        print(f"Pre-place pose: position={self.pre_place_pose[0]}, orientation={self.pre_place_pose[1]}")
        self._send_plan_request(self.pre_place_pose, "PRE_PLACE_IN_LOWER_DRAWER")

    def _request_place_motion(self):
        """Request motion from the pre-place pose to the lower-drawer target."""
        if self.grasp_pose is None:
            self.get_logger().error("No grasp pose is available; cannot request placement motion.")
            self.current_state = self.STATE_IDLE
            return

        if self.pre_place_pose is None:
            self.get_logger().error("No pre-place pose is available; cannot request placement motion.")
            self.current_state = self.STATE_IDLE
            return

        self.current_state = self.STATE_WAITING_FOR_PLACE

        # Keep the approach orientation established during pre-place. Reusing the
        # grasp-frame quaternion here makes the planner rotate the wrist during
        # the final translation into the drawer instead of moving forward.
        self.place_pose = (self.lower_drawer_position, self.pre_place_pose[1])
        print(f"Place pose: position={self.place_pose[0]}, orientation={self.place_pose[1]}")
        self._send_plan_request(self.place_pose, "PLACE_IN_LOWER_DRAWER")

    def _close_gripper_then_request_pre_place(self):
        """Give the gripper command one control cycle before transporting the cube."""
        if self.gripper_close_timer is not None:
            self.gripper_close_timer.cancel()
            self.gripper_close_timer = None

        self.get_logger().info("Gripper closed. Requesting lower-drawer pre-place motion.")
        self._request_pre_place_motion()

    def _send_plan_request(self, pose_tuple, motion_type):
        """Send a motion plan request to the motion planner."""
        position, orientation = pose_tuple

        request = PlanMotion.Request()
        request.grasp_pose.header = self.latest_cube_pose.header
        request.grasp_pose.pose.position.x = position[0]
        request.grasp_pose.pose.position.y = position[1]
        request.grasp_pose.pose.position.z = position[2]
        request.grasp_pose.pose.orientation.x = orientation[0]
        request.grasp_pose.pose.orientation.y = orientation[1]
        request.grasp_pose.pose.orientation.z = orientation[2]
        request.grasp_pose.pose.orientation.w = orientation[3]

        if not self.plan_motion_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/plan_motion service not available.")
            self.current_state = self.STATE_IDLE
            return

        future = self.plan_motion_client.call_async(request)
        future.add_done_callback(lambda f: self._plan_response_callback(f, motion_type))

        self.get_logger().info(
            f"Requested {motion_type} motion plan: frame_id={self.latest_cube_pose.header.frame_id}, "
            f"position={position}, orientation={orientation}"
        )

    def motion_completed_callback(self, msg):
        """Handle motion completion signal from MuJoCo."""
        if not msg.data:
            return  # Ignore False messages

        if self.current_state == self.STATE_WAITING_FOR_SETUP:
            self.get_logger().info("Setup motion completed. Requesting grasp motion.")
            self._request_grasp_motion()
        elif self.current_state == self.STATE_WAITING_FOR_GRASP:
            self.get_logger().info("Grasp motion completed. Closing gripper.")
            self.gripper_command_publisher.publish(Bool(data=True))  # Close the gripper
            self.current_state = self.STATE_WAITING_FOR_GRIPPER_CLOSE
            # The simulator applies gripper commands in its control loop. Wait
            # briefly before starting the trajectory that transports the cube.
            self.gripper_close_timer = self.create_timer(
                0.25, self._close_gripper_then_request_pre_place
            )
        elif self.current_state == self.STATE_WAITING_FOR_PRE_PLACE:
            self.get_logger().info("Pre-place motion completed. Requesting lower-drawer placement motion.")
            self._request_place_motion()
        elif self.current_state == self.STATE_WAITING_FOR_PLACE:
            self.get_logger().info("Lower-drawer placement motion completed. Opening gripper.")
            self.gripper_command_publisher.publish(Bool(data=False))  # Release the cube
            self.current_state = self.STATE_DONE
        else:
            # Ignore completion signals if not waiting for motion
            self.get_logger().debug(f"Received /motion_completed but not waiting for motion (state={self.current_state})")


    def _plan_response_callback(self, future, motion_type="UNKNOWN"):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"/plan_motion service call failed: {exc}")
            self.current_state = self.STATE_IDLE
            return

        self.get_logger().info(
            f"Plan response for {motion_type}: success={response.success}, message={response.message}, "
            f"trajectory_points={len(response.trajectory.points)}"
        )

        if response.success:
            self.get_logger().info(f"{motion_type} motion plan succeeded; waiting for execution to complete.")
        else:
            self.get_logger().warn(f"{motion_type} motion plan failed; resetting to idle state.")
            self.current_state = self.STATE_IDLE

def main(args=None):

    rclpy.init(args=args)

    node = TaskPlannerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
