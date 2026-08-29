import numpy as np
from scipy.spatial.transform import Rotation

from Panda_scene import PandaScene


def pose_to_matrix(position, quaternion):
    """
    Convert a MuJoCo pose to a 4x4 homogeneous transformation matrix.

    MuJoCo quaternion format:
        [w, x, y, z]

    scipy Rotation expects:
        [x, y, z, w]
    """

    T = np.eye(4)

    # MuJoCo -> scipy quaternion convention
    quat_xyzw = np.array([
        quaternion[1],
        quaternion[2],
        quaternion[3],
        quaternion[0],
    ])

    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position

    return T


def main():

    # ============================================================
    # Create the same Panda simulation
    # ============================================================

    env = PandaScene(
        robots=["Panda"],
        env_configuration="default",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    env.reset()


    # ============================================================
    # Get Panda bodies
    # ============================================================

    base_id = env.sim.model.body_name2id("robot0_link0")
    link7_id = env.sim.model.body_name2id("robot0_link7")


    # ============================================================
    # Get poses in WORLD frame
    # ============================================================

    base_position_world = env.sim.data.body_xpos[base_id].copy()
    base_quaternion_world = env.sim.data.body_xquat[base_id].copy()

    link7_position_world = env.sim.data.body_xpos[link7_id].copy()
    link7_quaternion_world = env.sim.data.body_xquat[link7_id].copy()


    # ============================================================
    # Convert to transformation matrices
    # ============================================================

    T_world_base = pose_to_matrix(
        base_position_world,
        base_quaternion_world
    )

    T_world_link7 = pose_to_matrix(
        link7_position_world,
        link7_quaternion_world
    )


    # ============================================================
    # Transform LINK7 -> PANDA BASE
    #
    # T_base_link7 =
    #     inverse(T_world_base) * T_world_link7
    # ============================================================

    T_base_link7 = (
        np.linalg.inv(T_world_base)
        @ T_world_link7
    )


    # ============================================================
    # Print results
    # ============================================================

    np.set_printoptions(
        precision=8,
        suppress=True
    )

    print()
    print("==========================================")
    print("      PANDA KINEMATICS TEST")
    print("==========================================")

    print()
    print("Panda base world:")
    print("Position:", base_position_world)
    print("Quaternion [w,x,y,z]:", base_quaternion_world)

    print()
    print("Panda link7 world:")
    print("Position:", link7_position_world)
    print("Quaternion [w,x,y,z]:", link7_quaternion_world)

    print()
    print("==========================================")
    print("LINK7 RELATIVE TO PANDA BASE")
    print("==========================================")

    print()
    print("Position:")
    print(T_base_link7[:3, 3])

    print()
    print("Rotation:")
    print(T_base_link7[:3, :3])

    print()
    print("Transformation matrix:")
    print(T_base_link7)

    print()
    print("==========================================")

    hand_id = env.sim.model.body_name2id("robot0_right_hand")

    position = env.sim.data.body_xpos[hand_id]
    quaternion = env.sim.data.body_xquat[hand_id]

    print("Hand position:", position)
    print("Hand quaternion [w,x,y,z]:", quaternion)

    env.close()


if __name__ == "__main__":
    main()