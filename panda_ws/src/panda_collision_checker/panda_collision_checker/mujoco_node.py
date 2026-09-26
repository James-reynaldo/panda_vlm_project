import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Bool

import numpy as np
import robosuite
from panda_collision_checker.Panda_scene import PandaScene

# Robosuite input handling for manual control
try:
    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action
    ROBOSUITE_AVAILABLE = True
except ImportError:
    ROBOSUITE_AVAILABLE = False


class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_node')

        # Declare and get control mode parameter
        self.declare_parameter('control_mode', 'planner')  # 'planner' or 'manual'
        self.control_mode = self.get_parameter('control_mode').value
        self.get_logger().info(f"Control mode: {self.control_mode}")

        # Manual input needs a controller supported by robosuite input2action.
        # Planner trajectories keep PandaScene defaulting to JOINT_POSITION.
        controller_configs = None
        if self.control_mode == "manual":
            controller_configs = robosuite.load_controller_config(
                default_controller="OSC_POSE"
            )

        # Create MuJoCo environment
        self.env = PandaScene(
            robots=["Panda"],
            env_configuration="default",
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            controller_configs=controller_configs,
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
        
        # Planner mode subscribers
        self.trajectory_subscriber = self.create_subscription(JointTrajectory, 'joint_trajectory', self.trajectory_callback, 10)
        
        self.gripper_command_subscriber = self.create_subscription(Bool, '/gripper_command', self.gripper_command_callback, 10)

        # Track trajectory execution state
        self._trajectory_completed_published = False
        self._gripper_closed = False  # Track gripper state
        self._gripper_command_received = False

        # Initialize input device for manual control mode (lazy initialization)
        self.input_device = None
        self.input_device_initialized = False
        
        # Single synchronized simulation/render/publish loop
        self.sim_timer = self.create_timer(0.01, self.simulation_loop)

    def print_gripper_orientation(self):
        """Print current gripper orientation as quaternion in xyzw order."""
        gripper_id = self.env.sim.model.body_name2id("robot0_right_hand")

        # MuJoCo stores quaternions as [w, x, y, z]
        quat_wxyz = self.env.sim.data.body_xquat[gripper_id]

        # Convert to [x, y, z, w]
        quat_xyzw = np.array([
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0]
        ])

        self.get_logger().info(
            f"Gripper quaternion (xyzw): "
            f"[{quat_xyzw[0]:.6f}, "
            f"{quat_xyzw[1]:.6f}, "
            f"{quat_xyzw[2]:.6f}, "
            f"{quat_xyzw[3]:.6f}]"
        )
        
    def _get_param_or_default(self, name, default):
        """Get parameter value, or return default if not set."""
        self.declare_parameter(name, default)
        return self.get_parameter(name).value
    
    def _initialize_input_device(self):
        """Lazy initialize input device on first simulation loop."""
        if self.input_device_initialized or self.control_mode != 'manual':
            return
        
        self.input_device_initialized = True
        
        if not ROBOSUITE_AVAILABLE:
            self.get_logger().error("robosuite not available. Cannot use manual control mode.")
            return
        
        device_type = self._get_param_or_default('device_type', 'keyboard')
        try:
            # if device_type == 'spacemouse':
            #     try:
            #         self.input_device = SpaceMouse()
            #         self.get_logger().info("Initialized SpaceMouse input device for manual control")
            #     except Exception as e:
            #         self.get_logger().warn(
            #             f"Failed to initialize SpaceMouse: {e}. "
            #             f"Falling back to keyboard input."
            #         )
            #         device_type = 'keyboard'
            
            if device_type == 'keyboard' and self.input_device is None:
                try:
                    self.input_device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
                    self.input_device.start_control()
                    self.get_logger().info("Initialized Keyboard input device for manual control")
                except Exception as kb_error:
                    self.get_logger().error(
                        f"Failed to initialize Keyboard device: {kb_error}. "
                        f"This typically means X11 display is not available. "
                        f"Make sure DISPLAY is set or use X11 forwarding: export DISPLAY=:0"
                    )
                    self.input_device = None
        except Exception as e:
            self.get_logger().error(f"Unexpected error during input device initialization: {e}")
            self.input_device = None

        # Joint position tolerance for trajectory completion (radians)
        self.position_tolerance = 0.06

        self.cube_pose_publisher = self.create_publisher(PoseStamped, 'cube_pose', 10)
        self.panda_base_pose_publisher = self.create_publisher(PoseStamped, 'panda_base_pose', 10)
        self.panda_joint_state_publisher = self.create_publisher(JointState, 'panda_joint_states', 10)
        self.motion_completed_publisher = self.create_publisher(Bool, '/motion_completed', 10)
        
        # Planner mode subscribers
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
        self._gripper_command_received = True
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
        # Lazy initialize input device on first loop
        self._initialize_input_device()
        
        action = self._zero_action.copy()

        # ---------------------------------------------------------
        # Control mode dispatch
        # ---------------------------------------------------------
        if self.control_mode == 'planner':
            action = self._control_planner_mode(action)
        elif self.control_mode == 'manual':
            action = self._control_manual_mode(action)
        else:
            self.get_logger().warn(f"Unknown control_mode: {self.control_mode}. Using zero action.")

        # ---------------------------------------------------------
        # Gripper
        # ---------------------------------------------------------
        # The final action slot is the gripper for both JOINT_POSITION and OSC_POSE.
        # In manual mode, preserve the keyboard gripper toggle unless ROS overrides it.
        if self.control_mode == "planner" or self._gripper_command_received:
            action[-1] = 1.0 if self._gripper_closed else -1.0

        # ---------------------------------------------------------
        # Simulation step
        # ---------------------------------------------------------
        q_before = np.asarray(
            self.env.robots[0]._joint_positions,
            dtype=np.float64
        ).copy()

        self.env.step(action)

        if self.control_mode == "manual":
            self.print_gripper_orientation()

        q_after = np.asarray(
            self.env.robots[0]._joint_positions,
            dtype=np.float64
        ).copy()

        # ---------------------------------------------------------
        # Rendering
        # ---------------------------------------------------------
        self.env.render()

        # ---------------------------------------------------------
        # Completion / state publishing
        # ---------------------------------------------------------
        if self.control_mode == 'planner':
            self._check_and_publish_trajectory_completion()

        self.publish_cube_pose()
        self.publish_panda_base_pose()
        self.publish_panda_joint_states()

    def _control_planner_mode(self, action):
        """Handle trajectory-based planner control mode."""
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

                # self.get_logger().info(
                #     f"[TRACKING] "
                #     f"joint={max_joint} "
                #     f"target={trajectory_goal[max_joint]:.6f} "
                #     f"current={current_qpos[max_joint]:.6f} "
                #     f"error={joint_delta[max_joint]:+.6f} "
                #     f"raw_action={action[max_joint]:+.6f} "
                #     f"scaled_action={scaled_action[max_joint]:+.6f}"
                # )

            except Exception as e:
                self.get_logger().warn(
                    f"[TRACKING] Could not inspect controller scaling: {e}"
                )

        return action

    def _control_manual_mode(self, action):
        """Handle manual input device control mode using robosuite's input2action."""
        if self.input_device is None:
            # No input device, use zero action
            return self._zero_action.copy()
        
        try:
            # input2action reads the device state and returns (action, grasp).
            robot = self.env.robots[0]
            device_action, _ = input2action(self.input_device, robot)
            if device_action is not None:
                action[:len(device_action)] = device_action
        except Exception as e:
            self.get_logger().warn(f"Error processing input: {e}")
            action = self._zero_action.copy()
        
        return action

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