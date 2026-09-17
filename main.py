from panda_vlm_project.panda_ws.src.panda_collision_checker.panda_collision_checker.Panda_scene import PandaScene
from robosuite.utils.input_utils import input2action
from robosuite.devices import Keyboard, SpaceMouse
import robosuite

env = PandaScene(
    robots=["Panda"],
    env_configuration="default",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    controller_configs=robosuite.load_controller_config(default_controller="OSC_POSE"),
)
# Choose your input device
device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
# device = SpaceMouse(env=env)

device.start_control()
# Reset the environment
obs = env.reset()
while True:
    # Get input from the user
    action, grasp = input2action(
            device=device,
            robot=env.robots[0],
        )
    if action is None:
        break

    obs, reward, done, info = env.step(action)
    # Render the current state
    env.render()
