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

    def check_configuration(self, joint_positions):

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

            name1 = self.env.sim.model.geom(contact.geom1).name
            name2 = self.env.sim.model.geom(contact.geom2).name
            

            print(f"Contact: {name1} <-> {name2}")

            geom1_name = self.env.sim.model.geom(
                contact.geom1
            ).name

            geom2_name = self.env.sim.model.geom(
                contact.geom2
            ).name

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
                print(
                    f"Collision: "
                    f"{geom1_name} <-> {geom2_name}"
                )
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
                print(
                    f"Collision: "
                    f"{geom1_name} <-> {geom2_name}"
                )
                return True

        return False