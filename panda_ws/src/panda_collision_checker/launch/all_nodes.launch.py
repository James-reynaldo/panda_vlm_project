from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='panda_collision_checker',
            executable='mujoco_node',
            name='mujoco_node',
            output='screen',
        ),
        Node(
            package='panda_collision_checker',
            executable='task_planner_node',
            name='task_planner_node',
            output='screen',
        ),
        Node(
            package='panda_collision_checker',
            executable='collision_checker_node',
            name='collision_checker_node',
            output='screen',
        ),
        Node(
            package='panda_planner',
            executable='motion_planner',
            name='motion_planner',
            output='screen',
        ),
    ])
