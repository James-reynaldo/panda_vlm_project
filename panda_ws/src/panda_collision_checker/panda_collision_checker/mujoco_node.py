import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Bool

import numpy as np
from panda_collision_checker.Panda_scene import PandaScene


class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_node')

        # Create MuJoCo environment
        self.env = PandaScene(
            robots=["Panda"],
            env_configuration="default",
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=False,
        )
        self.env.reset()

        self.cube_id = self.env.sim.model.body_name2id("red_cube_main")
        self._zero_action = np.zeros(self.env.action_dim, dtype=np.float64)
        self._trajectory = None
        self._trajectory_start = None
        self.get_logger().info(f"Cube ID: {self.cube_id}")

        # Joint position tolerance for trajectory completion (radians)
        self.position_tolerance = 0.06

        self.cube_pose_publisher = self.create_publisher(PoseStamped, 'cube_pose', 10)
        self.panda_base_pose_publisher = self.create_publisher(PoseStamped, 'panda_base_pose', 10)
        self.panda_joint_state_publisher = self.create_publisher(JointState, 'panda_joint_states', 10)
        self.motion_completed_publisher = self.create_publisher(Bool, '/motion_completed', 10)
        self.trajectory_subscriber = self.create_subscription(JointTrajectory, 'joint_trajectory', self.trajectory_callback, 10)
        self.gripper_command_subscriber = self.create_subscription(Bool, '/gripper_command', self.gripper_command_callback, 10)

        # Track trajectory execution state
        self._trajectory_completed_published = False
        self._gripper_closed = False  # Track gripper state

        # Single synchronized simulation/render/publish loop
        self.sim_timer = self.create_timer(0.05, self.simulation_loop)

    def trajectory_callback(self, msg):
        if not msg.points or not msg.joint_names:
            return

        points = []
        for point in msg.points:
            if len(point.positions) == 0:
                continue
            t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            points.append((float(t), np.asarray(point.positions, dtype=np.float64)))

        if not points:
            return

        self._trajectory = points
        self._trajectory_start = self.get_clock().now().nanoseconds * 1e-9
        self._trajectory_completed_published = False  # Reset flag for new trajectory
        self.get_logger().info(f"Received trajectory with {len(points)} points. Target duration: {points[-1][0]:.3f}s")

    def gripper_command_callback(self, msg):
        """Handle gripper open/close command."""
        self._gripper_closed = msg.data
        self.get_logger().info(f"Gripper command received: {'CLOSE' if msg.data else 'OPEN'}")

    def _interpolate_trajectory(self):
        if self._trajectory is None or self._trajectory_start is None:
            return None

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._trajectory_start

        if elapsed <= self._trajectory[0][0]:
            target_qpos = self._trajectory[0][1].copy()
        elif elapsed >= self._trajectory[-1][0]:
            target_qpos = self._trajectory[-1][1].copy()
        else:
            for i in range(1, len(self._trajectory)):
                t_prev, q_prev = self._trajectory[i - 1]
                t_next, q_next = self._trajectory[i]
                if elapsed <= t_next:
                    if t_next <= t_prev:
                        target_qpos = q_next.copy()
                    else:
                        alpha = (elapsed - t_prev) / (t_next - t_prev)
                        target_qpos = q_prev + alpha * (q_next - q_prev)
                    break
            else:
                target_qpos = self._trajectory[-1][1].copy()

        return target_qpos

    def simulation_loop(self):
        action = self._zero_action.copy()

        # ---------------------------------------------------------
        # Arm trajectory
        # ---------------------------------------------------------
        trajectory_goal = self._interpolate_trajectory()

        if trajectory_goal is not None:
            robot = self.env.robots[0]
            controller = robot.controller

            current_qpos = np.asarray(
                robot._joint_positions,
                dtype=np.float64
            )

            joint_delta = trajectory_goal - current_qpos
            action_dim = controller.control_dim

            # IMPORTANT:
            # action is the normalized JOINT_POSITION controller input.
            # Do NOT manually convert it using output/input ranges here.
            action[:action_dim] = joint_delta[:action_dim]

            # -----------------------------------------------------
            # Controller diagnostic
            # -----------------------------------------------------
            try:
                scaled_action = controller.scale_action(
                    action[:action_dim].copy()
                )

                max_joint = int(np.argmax(np.abs(joint_delta[:action_dim])))

                self.get_logger().info(
                    f"[TRACKING] "
                    f"joint={max_joint} "
                    f"target={trajectory_goal[max_joint]:.6f} "
                    f"current={current_qpos[max_joint]:.6f} "
                    f"error={joint_delta[max_joint]:+.6f} "
                    f"raw_action={action[max_joint]:+.6f} "
                    f"scaled_action={scaled_action[max_joint]:+.6f}"
                )

            except Exception as e:
                self.get_logger().warn(
                    f"[TRACKING] Could not inspect controller scaling: {e}"
                )

        # ---------------------------------------------------------
        # Gripper
        # ---------------------------------------------------------
        if self._gripper_closed:
            action[7] = 1.0
        else:
            action[7] = -1.0

        # ---------------------------------------------------------
        # Simulation step
        # ---------------------------------------------------------
        q_before = np.asarray(
            self.env.robots[0]._joint_positions,
            dtype=np.float64
        ).copy()

        self.env.step(action)

        q_after = np.asarray(
            self.env.robots[0]._joint_positions,
            dtype=np.float64
        ).copy()

        # ---------------------------------------------------------
        # Actual movement diagnostic
        # ---------------------------------------------------------
        if trajectory_goal is not None:
            actual_change = q_after - q_before

            max_joint = int(np.argmax(np.abs(joint_delta[:action_dim])))

            self.get_logger().info(
                f"[TRACKING AFTER] "
                f"joint={max_joint} "
                f"q_before={q_before[max_joint]:.6f} "
                f"q_after={q_after[max_joint]:.6f} "
                f"actual_change={actual_change[max_joint]:+.6f}"
            )

        # ---------------------------------------------------------
        # Rendering
        # ---------------------------------------------------------
        self.env.render()

        # ---------------------------------------------------------
        # Completion / state publishing
        # ---------------------------------------------------------
        self._check_and_publish_trajectory_completion()

        self.publish_cube_pose()
        self.publish_panda_base_pose()
        self.publish_panda_joint_states()

    def _check_and_publish_trajectory_completion(self):
        """Check if trajectory has finished and robot has reached target position within tolerance."""
        if self._trajectory is None or self._trajectory_start is None:
            return

        if self._trajectory_completed_published:
            return  # Already published for this trajectory

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._trajectory_start
        trajectory_duration = self._trajectory[-1][0]

        # Only check for completion after trajectory time has elapsed
        if elapsed >= trajectory_duration:
            # Get actual joint positions from robot
            robot = self.env.robots[0]
            actual_qpos = np.asarray(robot._joint_positions, dtype=np.float64)
            target_qpos = self._trajectory[-1][1]
            
            # Calculate absolute joint position errors
            joint_errors = np.abs(actual_qpos - target_qpos)
            max_error = np.max(joint_errors)
            
            # Check if within tolerance
            if max_error < self.position_tolerance:
                completion_msg = Bool(data=True)
                self.motion_completed_publisher.publish(completion_msg)
                self._trajectory_completed_published = True
                self.get_logger().info(
                    f"Trajectory execution completed. Final max joint position error: {max_error:.6f} rad "
                    f"(tolerance: {self.position_tolerance} rad). Published /motion_completed."
                )
                # Clear trajectory to prevent re-publishing
                self._trajectory = None
            else:
                # Still settling; keep commanding final position (done in _interpolate_trajectory)
                self.get_logger().debug(
                    f"Robot settling towards target. Max joint error: {max_error:.6f} rad "
                    f"(tolerance: {self.position_tolerance} rad)"
                )

    def publish_cube_pose(self):
        # Example pose data; replace with actual data from MuJoCo

        # Get cube position 
        position = self.env.sim.data.body_xpos[self.cube_id]
        orientation = self.env.sim.data.body_xquat[self.cube_id]
        # quarternion = self.rotation_matrix_to_quaternion(orientation)
        quarternion = orientation  # Assuming orientation is already in quaternion format

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.x = quarternion[0]
        pose_msg.pose.orientation.y = quarternion[1]
        pose_msg.pose.orientation.z = quarternion[2]
        pose_msg.pose.orientation.w = quarternion[3]

        self.cube_pose_publisher.publish(pose_msg)

    def publish_panda_base_pose(self):
        # Get Panda base position and orientation
        panda_base_id = self.env.sim.model.body_name2id("robot0_link0")
        position = self.env.sim.data.body_xpos[panda_base_id]
        orientation = self.env.sim.data.body_xquat[panda_base_id]
        quarternion = orientation  # Assuming orientation is already in quaternion format

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.x = quarternion[0]
        pose_msg.pose.orientation.y = quarternion[1]
        pose_msg.pose.orientation.z = quarternion[2]
        pose_msg.pose.orientation.w = quarternion[3]

        self.panda_base_pose_publisher.publish(pose_msg)

    def publish_panda_joint_states(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = "world"
        joint_state_msg.name = [f"robot0_joint{i+1}" for i in range(7)]
        joint_state_msg.position = self.env.sim.data.qpos[:7].tolist()  # Assuming the first 7 qpos are the joint positions

        self.panda_joint_state_publisher.publish(joint_state_msg)
    @staticmethod
    def rotation_matrix_to_quaternion(R):

        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)

            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s

        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:

            s = 2.0 * np.sqrt(
                1.0 + R[0, 0] - R[1, 1] - R[2, 2]
            )

            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s

        elif R[1, 1] > R[2, 2]:

            s = 2.0 * np.sqrt(
                1.0 + R[1, 1] - R[0, 0] - R[2, 2]
            )

            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s

        else:

            s = 2.0 * np.sqrt(
                1.0 + R[2, 2] - R[0, 0] - R[1, 1]
            )

            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([x, y, z, w])

def main(args=None):
    rclpy.init(args=args)
    mujoco_node = MujocoNode()
    rclpy.spin(mujoco_node)
    mujoco_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()