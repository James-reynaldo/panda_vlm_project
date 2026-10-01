import numpy as np


class CollisionChecker:

    # Geometries belonging to the Panda that should participate
    # in environment collision checking.
    PANDA_GEOMS = {
        "robot0_link0_collision",
        "robot0_link1_collision",
        "robot0_link2_collision",
        "robot0_link3_collision",
        "robot0_link4_collision",
        "robot0_link5_collision",
        "robot0_link6_collision",
        "robot0_link7_collision",
        "gripper0_hand_collision",
        "gripper0_finger1_collision",
        "gripper0_finger1_pad_collision",
        "gripper0_finger2_collision",
        "gripper0_finger2_pad_collision",
    }

    ROBOT_HAND_BODIES = {
        "robot0_right_hand",
        "gripper0_right_gripper",
        "gripper0_eef",
        "gripper0_leftfinger",
        "gripper0_finger_joint1_tip",
        "gripper0_rightfinger",
        "gripper0_finger_joint2_tip",
    }

    CUBE_BODIES = {
        "red_cube_main",
    }

    TABLE_GEOMS = {
        "table_collision",
    }

    CABINET_GEOMS = {
        "cabinet_back",
        "cabinet_left",
        "cabinet_right",
        "cabinet_bottom",
        "cabinet_top",
        "cabinet_shelf_1",
    }

    def __init__(self, env):
        self.env = env

    def _geom_body_name(self, geom_id):
        body_id = self.env.sim.model.geom_bodyid[geom_id]
        return self.env.sim.model.body_id2name(body_id)

    def _has_hand_cube_contact(self, geom1_name, geom2_name):
        return (
            (geom1_name in self.ROBOT_HAND_BODIES and geom2_name in self.CUBE_BODIES)
            or
            (geom2_name in self.ROBOT_HAND_BODIES and geom1_name in self.CUBE_BODIES)
        )

    def check_configuration(self, joint_positions, red_cube_collision_enabled=True):

        joint_positions = np.asarray(
            joint_positions,
            dtype=float
        )

        if joint_positions.shape != (7,):
            raise ValueError(
                f"Expected 7 joint positions, "
                f"got {joint_positions.shape}"
            )

        # Set Panda joint configuration
        self.env.sim.data.qpos[
            self.env.robots[0]._ref_joint_pos_indexes
        ] = joint_positions

        # Recompute kinematics and contacts
        self.env.sim.forward()

        # Check all contacts
        for i in range(self.env.sim.data.ncon):

            contact = self.env.sim.data.contact[i]

            geom1_name = self._geom_body_name(contact.geom1)
            geom2_name = self._geom_body_name(contact.geom2)

            if red_cube_collision_enabled and self._has_hand_cube_contact(geom1_name, geom2_name):
                return True

            # Panda <-> table
            if (
                (
                    geom1_name in self.PANDA_GEOMS
                    and geom2_name in self.TABLE_GEOMS
                )
                or
                (
                    geom2_name in self.PANDA_GEOMS
                    and geom1_name in self.TABLE_GEOMS
                )
            ):
                # print(
                #     f"Collision: "
                #     f"{geom1_name} <-> {geom2_name}"
                # )
                return True

            # Panda <-> cabinet
            if (
                (
                    geom1_name in self.PANDA_GEOMS
                    and geom2_name in self.CABINET_GEOMS
                )
                or
                (
                    geom2_name in self.PANDA_GEOMS
                    and geom1_name in self.CABINET_GEOMS
                )
            ):
                # print(
                #     f"Collision: "
                #     f"{geom1_name} <-> {geom2_name}"
                # )
                return True
        # print("No collision detected.")
        return False