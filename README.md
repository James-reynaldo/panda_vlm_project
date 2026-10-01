colcon build
ros2 run panda_collision_checker mujoco_node 
ros2 run panda_collision_checker task_planner_node
ros2 run panda_collision_checker collision_checker_node 
ros2 run panda_planner motion_planner 

ros2 run panda_collision_checker mujoco_node --ros-args -p control_mode:=manual -p device_type:=keyboard
ros2 launch panda_collision_checker all_nodes.launch.py
