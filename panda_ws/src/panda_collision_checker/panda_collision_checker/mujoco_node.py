import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

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

        self.cube_pose_publisher = self.create_publisher(PoseStamped, 'cube_pose', 10)
        self.panda_base_pose_publisher = self.create_publisher(PoseStamped, 'panda_base_pose', 10)
        self.panda_joint_state_publisher = self.create_publisher(JointState, 'panda_joint_states', 10)
        self.trajectory_subscriber = self.create_subscription(JointTrajectory, 'joint_trajectory', self.trajectory_callback, 10)

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
        trajectory_goal = self._interpolate_trajectory()

        if trajectory_goal is not None:
            robot = self.env.robots[0]
            current_qpos = np.asarray(robot._joint_positions, dtype=np.float64)
            joint_delta = np.asarray(trajectory_goal, dtype=np.float64) - current_qpos
            action_dim = robot.controller.control_dim
            action[:action_dim] = joint_delta[:action_dim]

        # 1) Advance the MuJoCo/robosuite simulation by one timestep.
        self.env.step(action)

        # 2) Render the environment.
        self.env.render()

        # 3-5) Publish state in the same loop.
        self.publish_cube_pose()
        self.publish_panda_base_pose()
        self.publish_panda_joint_states()

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