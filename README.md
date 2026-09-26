colcon build
ros2 run panda_collision_checker mujoco_node 
ros2 run panda_collision_checker task_planner_node
ros2 run panda_collision_checker collision_checker_node 
ros2 run panda_planner motion_planner 
# Trajectory-based (planner) - DEFAULT
ros2 launch panda_collision_checker mujoco_node.launch.py

# Manual control (input2action)
ros2 launch panda_collision_checker mujoco_node.launch.py control_mode:=manual

# Or via command line
ros2 run panda_collision_checker mujoco_node --ros-args -p control_mode:=manual -p device_type:=spacemouse

rm -rf ~/.ros/log/*