from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject
from robosuite.utils.placement_samplers import UniformRandomSampler
import numpy as np
import os

class CabinetObject(MujocoXMLObject):
    """
    A cabinet object that can be used in the environment.
    """

    def __init__(self, name):
        super().__init__(
            fname=os.path.join(os.path.dirname(__file__), "assets", "cabinet.xml"),
            name=name,
            joints=[
                {
                    "type": "free",
                    "name": "free_joint",
                }
            ],
            obj_type="all"
        )

class PandaScene(SingleArmEnv):
    """
    A manipulation environment intended for a single robot arm.
    """

    def __init__(self, robots, env_configuration="default", render=False,has_offscreen_renderer=False,controller_configs=None, **kwargs):
        self.table_full_size = (1.0, 1.0, 0.05)
        self.red_cube_size = (0.02, 0.02, 0.1)
        self.table_offset = (0, 0, 0.80)
        self.object_placements = None

        self.cabinet_pos = np.array([0.0, 0.3, 0.82])
        self.cabinet_quat = np.array([1.0, 0.0, 0.0, 0.0])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            control_freq=20,
            horizon=1000,
            ignore_done=True,
            hard_reset=False,
            camera_names=None,
            camera_heights=None,
            camera_widths=None,
            camera_depths=None,
        )
        

    def reward(self, action=None):
        """
        Reward function for the environment. This is a placeholder and should be implemented based on the specific task.
        """
        # Placeholder reward: return 0 for now
        return 0.0

    def _load_model(self):
        """
        Loads an arena and a task into the environment.
        """
        super()._load_model()
        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Get the Panda MJCF models
        robot1 = self.robots[0].robot_model

        robot1.set_base_xpos((0, -0.65, 0))

        # rotate the robot to face the table
        robot1.set_base_ori((0, 0, np.pi/2))

        # print("Panda 1 table position:",
        #     robot1.base_xpos_offset["table"](self.table_full_size[0]))

        # # Position Panda 1
        # xpos = np.array(robot1.base_xpos_offset["table"](
        #     self.table_full_size[0]
        # ))
        # robot1.set_base_xpos(xpos)

        # # Position Panda 2 on the opposite side
        # xpos2 = xpos.copy()
        # xpos2[1] = -xpos2[1]
        # robot2.set_base_xpos(xpos2)
        red_cube = BoxObject(
            name="red_cube",
            size_min=self.red_cube_size,
            size_max=self.red_cube_size,
            rgba=[1, 0, 0, 1],
        )

        cabinet = CabinetObject(
            name="cabinet",
        )
        
        # Load the task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot1],
            mujoco_objects=[red_cube, cabinet],
        )

        # self.object_placement_initializer = UniformRandomSampler(
        #     name="object_sampler",
        #     mujoco_objects=self.model.mujoco_objects,
        #     x_range=[-0.3, 0.3],
        #     y_range=[-0.3, 0.3],
        #     rotation=None,
        #     ensure_object_boundary_in_range=False,
        #     ensure_valid_placement=True,
        #     reference_pos=self.table_offset,
        # )
        # self.object_placements = self.object_placement_initializer.sample()


    def _reset_internal(self):
        """
        Resets the internal state of the environment.
        """
        super()._reset_internal()

        red_cube = self.model.mujoco_objects[0]
        cabinet = self.model.mujoco_objects[1]

        red_cube_pos = np.array([-0.2, 0.0, red_cube.size[2] / 2 + self.table_offset[2]])
        red_cube_quat = np.array([1, 0, 0, 0])

        self.sim.data.set_joint_qpos(red_cube.joints[0], np.concatenate([red_cube_pos, red_cube_quat]))
        self.sim.data.set_joint_qpos(cabinet.joints[0], np.concatenate([self.cabinet_pos, self.cabinet_quat]))
