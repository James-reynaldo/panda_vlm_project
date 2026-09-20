import numpy as np
from scipy.spatial.transform import Rotation


class TaskPlanner:

    def __init__(self, grasp_offset=0.10):
        """
        Parameters
        ----------
        grasp_offset : float
            Distance above the cube center at which the
            gripper grasp pose is generated.
        """
        self.grasp_offset = grasp_offset

    def generate_grasp_pose(self, cube_position, cube_orientation):
        """
        Generate a top-down grasp pose for the cube.

        Parameters
        ----------
        cube_position : array-like
            Cube position [x, y, z].
        cube_orientation : array-like
            Cube orientation as quaternion [x, y, z, w].

        Returns
        -------
        position : np.ndarray
            Desired gripper position [x, y, z].

        orientation : np.ndarray
            Desired gripper orientation as quaternion [x, y, z, w].
        """

        cube_position = np.asarray(
            cube_position,
            dtype=float
        )

        if cube_position.shape != (3,):
            raise ValueError(
                f"Expected cube position shape (3,), "
                f"got {cube_position.shape}"
            )

        cube_orientation = np.asarray(
            cube_orientation,
            dtype=float
        )

        if cube_orientation.shape != (4,):
            raise ValueError(
                f"Expected cube orientation shape (4,), "
                f"got {cube_orientation.shape}"
            )

        # Orientation facing front of cube
        cube_rotation = Rotation.from_quat(cube_orientation)
        side_rotation = Rotation.from_euler("xyz", [0, np.pi / 2, np.pi / 2])
        desired_rotation = cube_rotation * side_rotation
        desired_orientation = desired_rotation.as_quat()
        # Position above the cube
        grasp_offset = np.array([0.0, 0.0, self.grasp_offset])
        grasp_position = (cube_position + desired_rotation.apply(grasp_offset))


        # # TODO: verify Panda gripper orientation convention
        # # For now, use a fixed top-down orientation.
        # orientation = Rotation.from_euler(
        #     "xyz",
        #     [np.pi, 0.0, 0.0]
        # ).as_quat()

        return grasp_position, desired_orientation

    def generate_approach_pose(self, target_position, target_orientation):
        """Generate a pose offset from a target along its local approach axis.

        Unlike :meth:`generate_grasp_pose`, this preserves the supplied target
        orientation. It is suitable for approach poses such as pre-place.
        """
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,):
            raise ValueError(
                f"Expected target position shape (3,), got {target_position.shape}"
            )

        target_orientation = np.asarray(target_orientation, dtype=float)
        if target_orientation.shape != (4,):
            raise ValueError(
                f"Expected target orientation shape (4,), got {target_orientation.shape}"
            )

        target_rotation = Rotation.from_quat(target_orientation)
        approach_offset = np.array([0.0, 0.0, self.grasp_offset])
        approach_position = target_position + target_rotation.apply(approach_offset)

        return approach_position, target_orientation.copy()
