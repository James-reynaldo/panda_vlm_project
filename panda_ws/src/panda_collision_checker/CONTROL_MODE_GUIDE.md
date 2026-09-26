# MuJoCo Node Control Mode Guide

## Overview
The `mujoco_node` now supports two control modes:
1. **Planner Mode** (default): Uses trajectory-based control via JointTrajectory messages
2. **Manual Mode**: Uses real-time input2action from robosuite (Keyboard or SpaceMouse)

## Requirements for Manual Mode
Manual mode requires robosuite to be installed:
```bash
pip install robosuite
```

## Launching with Different Modes

### Planner Mode (Default)
This mode uses trajectory planning. Launch with:
```bash
ros2 launch panda_collision_checker mujoco_node.launch.py
# or explicitly
ros2 launch panda_collision_checker mujoco_node.launch.py control_mode:=planner
```

**Topics:**
- Subscribe to: `joint_trajectory` (JointTrajectory messages)
- Publish: `motion_completed` (Bool) when trajectory execution completes

### Manual Mode with Keyboard
Real-time control via keyboard input using robosuite's input2action:
```bash
ros2 launch panda_collision_checker mujoco_node.launch.py control_mode:=manual device_type:=keyboard
```

**Keyboard Controls** (typical):
- WASD: Move end-effector in X-Y plane
- Q/E: Move up/down (Z-axis)
- Arrow Keys: Rotate end-effector
- Spacebar: Toggle gripper open/close

### Manual Mode with SpaceMouse
Real-time control via SpaceMouse (3D mouse):
```bash
ros2 launch panda_collision_checker mujoco_node.launch.py control_mode:=manual device_type:=spacemouse
```

**No trajectory completion detection** — continuous teleoperation mode.

## Node Parameters

Set parameters in launch file or command line:
- `control_mode` (string): `'planner'` or `'manual'` (default: `'planner'`)
- `device_type` (string): `'keyboard'` or `'spacemouse'` (default: `'keyboard'`, only used in manual mode)

### Command Line Examples:
```bash
# Manual mode with keyboard (default)
ros2 run panda_collision_checker mujoco_node --ros-args -p control_mode:=manual

# Manual mode with SpaceMouse
ros2 run panda_collision_checker mujoco_node --ros-args -p control_mode:=manual -p device_type:=spacemouse
```

## Gripper Control

Both modes use the same gripper interface:
- Subscribe to: `/gripper_command` (Bool)
  - `True`: Close gripper
  - `False`: Open gripper

In manual mode, gripper control is typically integrated with the input device.

## Implementation Details

### Planner Mode
- Receives trajectory waypoints with timestamps via ROS topic
- Interpolates between waypoints in real-time
- Publishes `motion_completed` when trajectory execution completes within tolerance (0.06 rad)
- Uses position control with the robot's built-in controller

### Manual Mode
- Reads input from Keyboard or SpaceMouse device
- Converts raw device input to end-effector velocity/orientation commands using robosuite's `input2action`
- Applies actions directly to the simulator in real-time
- No trajectory completion detection
- Suitable for teleoperation and interactive testing

## Integration with run_demo.py

The manual mode implementation mirrors the `MANUAL_CONTROL` logic in `run_demo.py`:
- Same device initialization (Keyboard or SpaceMouse)
- Same `input2action` conversion from robosuite
- Same OSC_POSE controller interface

## Troubleshooting

**"robosuite not available" error in manual mode:**
- Install robosuite: `pip install robosuite`
- Verify installation: `python -c "from robosuite.devices import Keyboard; print('OK')"`

**Input device not responding:**
- Ensure X11 display is available (on SSH/headless systems, may need X11 forwarding)
- Try running with `export DISPLAY=:0` (Linux)
- Check device permissions (SpaceMouse may need udev rules)

**Action doesn't match expected controller:**
- Verify controller type in `_control_manual_mode()` method matches your robot setup
- Default is `OSC_POSE` (end-effector position/orientation control)

## Switching Modes at Runtime

To switch modes, relaunch with a different parameter:
```bash
# Switch from planner to manual
ros2 launch panda_collision_checker mujoco_node.launch.py control_mode:=manual

# Switch back to planner
ros2 launch panda_collision_checker mujoco_node.launch.py control_mode:=planner
```

## Code Example: Publishing Trajectories in Planner Mode

```python
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rospy
from rospy import Duration

pub = rospy.Publisher('joint_trajectory', JointTrajectory, queue_size=10)

traj = JointTrajectory()
traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']

# Add waypoint
point = JointTrajectoryPoint()
point.positions = [0, 0, 0, -1.57, 0, 1.57, 0.785]
point.time_from_start = Duration(5)  # 5 seconds
traj.points.append(point)

pub.publish(traj)
```
