import os
import numpy as np

from panda_collision_checker.Panda_scene import PandaScene
from panda_collision_checker.collision_checker import CollisionChecker
from scipy.spatial.transform import Rotation
def pose_to_matrix(position, quaternion_wxyz):
    """
    Convert a MuJoCo pose to a 4x4 homogeneous transformation matrix.

    MuJoCo quaternion format:
        [w, x, y, z]

    SciPy expects:
        [x, y, z, w]
    """

    w, x, y, z = quaternion_wxyz

    rotation = Rotation.from_quat([x, y, z, w])

    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = position

    return T

def main():

    # --------------------------------------------------
    # Configuration to test
    # --------------------------------------------------
    
    q = np.array([
        -0.000000, 0.196350, -0.000000, -2.617994, 0.000000, 2.941593, 0.785398
    ])

    # --------------------------------------------------
    # Create MuJoCo scene
    # --------------------------------------------------

    env = PandaScene(
        robots="Panda",
        render=False,
        has_offscreen_renderer=True,
    )

    env.reset()

    #print body names
    print("\n=== BODY NAMES ===")
    print(env.sim.model.body_names)

    # --------------------------------------------------
    # Collision check
    # --------------------------------------------------
    print("\n=== PANDA COLLISION MASKS ===")

    print("\n=== PANDA BODY FRAMES ===")

    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i)

        if "robot0" in name:
            print(
                f"{name}: "
                f"pos={env.sim.data.body_xpos[i]}, "
                f"quat={env.sim.data.body_xquat[i]}"
            )

    for name in CollisionChecker.PANDA_GEOMS:

        geom_id = env.sim.model.geom(name).id

        print(
            f"{name}: "
            f"contype={env.sim.model.geom_contype[geom_id]}, "
            f"conaffinity={env.sim.model.geom_conaffinity[geom_id]}"
        )
        checker = CollisionChecker(env)
    print("\n=== GEOMETRY POSITIONS ===")

    for name in [
        "robot0_link0_collision",
        "robot0_link1_collision",
        "robot0_link2_collision",
        "robot0_link3_collision",
        "robot0_link4_collision",
        "robot0_link5_collision",
        "robot0_link6_collision",
        "robot0_link7_collision",
        "cabinet_left",
        "cabinet_right",
        "cabinet_back",
    ]:

        geom_id = env.sim.model.geom(name).id

        print(
            f"{name}: "
            f"pos={env.sim.data.geom_xpos[geom_id]}, "
            f"size={env.sim.model.geom_size[geom_id]}"
        )
    collision = checker.check_configuration(q)

    print("Joint configuration:")
    print(q)

    print()
    print("Collision:", collision)

    base_id = env.sim.model.body_name2id("robot0_link0")
    hand_id = env.sim.model.body_name2id("robot0_right_hand")

    base_pos = env.sim.data.body_xpos[base_id]
    base_quat = env.sim.data.body_xquat[base_id]


    hand_pos = env.sim.data.body_xpos[hand_id]
    hand_quat = env.sim.data.body_xquat[hand_id]

    T_world_base = pose_to_matrix(
        base_pos,
        base_quat
    )

    print("\n=== PANDA BASE ===")
    print("Position:", base_pos)
    print("Quaternion [w,x,y,z]:", base_quat)

    T_world_hand = pose_to_matrix(
        hand_pos,
        hand_quat
    )
    print("\n=== PANDA HAND ===")
    print("Position:", hand_pos)
    print("Quaternion [w,x,y,z]:", hand_quat)

    T_base_hand = (
        np.linalg.inv(T_world_base)
        @ T_world_hand
    )
    print("\n=== HAND RELATIVE TO PANDA BASE ===")
    print("Position:", T_base_hand[:3, 3])
    print("Rotation:")
    print(T_base_hand[:3, :3])
    

    # --------------------------------------------------
    # Render the configuration
    # --------------------------------------------------

    # Make sure MuJoCo has updated the scene
    env.sim.forward()

    # Render from the default camera
    image = env.sim.render(
        camera_name="frontview",
        width=1000,
        height=800,
    )

    # --------------------------------------------------
    # Save image
    # --------------------------------------------------

    output_path = os.path.abspath("collision_test.png")

    # MuJoCo returns RGB, while image libraries typically
    # expect RGB as well, so this can be saved directly.
    from PIL import Image

    # save flipped image
    img = Image.fromarray(image)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(output_path)

    print()
    print("Image saved to:")
    print(output_path)

    env.close()


if __name__ == "__main__":
    main()