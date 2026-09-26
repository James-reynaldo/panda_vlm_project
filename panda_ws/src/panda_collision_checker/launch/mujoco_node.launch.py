from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='planner',
        description='Control mode for mujoco_node: "planner" (trajectory-based) or "manual" (input2action-based)',
        choices=['planner', 'manual']
    )
    
    device_type_arg = DeclareLaunchArgument(
        'device_type',
        default_value='keyboard',
        description='Input device for manual mode: "keyboard" or "spacemouse"',
        choices=['keyboard', 'spacemouse']
    )

    # Create mujoco_node with control_mode and device_type parameters
    mujoco_node = Node(
        package='panda_collision_checker',
        executable='mujoco_node',
        name='mujoco_node',
        parameters=[
            {'control_mode': LaunchConfiguration('control_mode')},
            {'device_type': LaunchConfiguration('device_type')},
        ],
        output='screen',
    )

    return LaunchDescription([
        control_mode_arg,
        device_type_arg,
        mujoco_node,
    ])
