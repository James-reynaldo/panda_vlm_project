#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>   
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "panda_planner/Eigen/Dense"
#include "panda_planner/Eigen/Geometry"

#include "panda_planner/franka_ik_He.hpp"

constexpr double HAND_OFFSET = -0.107;

static constexpr double PANDA_BASE_X = 0.0;
static constexpr double PANDA_BASE_Y = -0.65;
static constexpr double PANDA_BASE_Z = 0.912;

static constexpr double PANDA_BASE_QW = 0.70710678;
static constexpr double PANDA_BASE_QX = 0.0;
static constexpr double PANDA_BASE_QY = 0.0;
static constexpr double PANDA_BASE_QZ = 0.70710678;

Eigen::Isometry3d transformToPandaBase(
    const Eigen::Vector3d& position_world,
    const Eigen::Quaterniond& orientation_world,
    const Eigen::Vector3d& panda_base_position_world,
    const Eigen::Quaterniond& panda_base_orientation_world,
    double hand_offset = -0.107)
{
    // Create Panda base transformation in world frame
    Eigen::Isometry3d T_world_base =
        Eigen::Isometry3d::Identity();

    T_world_base.translation() =
        panda_base_position_world;

    T_world_base.linear() =
        panda_base_orientation_world.toRotationMatrix();

    // Create grasp transformation in world frame
    Eigen::Isometry3d T_world_grasp =
        Eigen::Isometry3d::Identity();

    T_world_grasp.translation() =
        position_world;

    T_world_grasp.linear() =
        orientation_world.toRotationMatrix();

    // WORLD -> PANDA BASE
    //
    // T_base_grasp =
    //     T_world_base.inverse()
    //     * T_world_grasp

    Eigen::Isometry3d T_base_grasp =
        T_world_base.inverse() *
        T_world_grasp;

    // ------------------------------------------------------------
    // Compensate for hand / EE offset
    //
    // Offset is along the gripper's local +Z axis.
    // col(2) expresses that axis in the Panda base frame.
    // ------------------------------------------------------------

    Eigen::Vector3d gripper_z_base =
        T_base_grasp.linear().col(2);

    T_base_grasp.translation() -=
        hand_offset * gripper_z_base;

    return T_base_grasp;
}


// ============================================================
// Convert Eigen transformation to IK array
// ============================================================

std::array<double, 16> toIKMatrix(
    const Eigen::Isometry3d& T)
{
    std::array<double, 16> result;

    // IK solver expects column-major order
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            result[col * 4 + row] =
                T.matrix()(row, col);
        }
    }

    return result;
}

bool solvePandaIK(
    const Eigen::Isometry3d& T_base_grasp,
    const std::array<double, 7>& q_actual,
    std::vector<std::array<double, 7>>& q_solution)
{
    std::array<double, 16> O_T_EE =
        toIKMatrix(T_base_grasp);

    // Search over redundant joint q7
    for (double q7 = -3.0; q7 <= 3.0; q7 += 0.25)
    {
        auto solutions = franka_IK_EE(
            O_T_EE,
            q7,
            q_actual
        );

        for (const auto& solution : solutions)
        {
            // Check whether solution is valid
            bool valid = true;

            for (double q : solution)
            {
                if (std::isnan(q) || std::isinf(q))
                {
                    valid = false;
                    break;
                }
            }

            if (!valid)
                continue;

            q_solution.push_back(solution);
            return true;
        }
    }

    return false;
}

class MotionPlanner : public rclcpp::Node
{
public:
    MotionPlanner() : Node("motion_planner")
    {      
        T_world_base_.setIdentity();
        T_world_base_.translation() = Eigen::Vector3d(PANDA_BASE_X, PANDA_BASE_Y, PANDA_BASE_Z);
        T_world_base_.linear() = Eigen::Quaterniond(PANDA_BASE_QW, PANDA_BASE_QX, PANDA_BASE_QY, PANDA_BASE_QZ).toRotationMatrix();

        grasp_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "grasp_pose", 10, std::bind(&MotionPlanner::graspCallback, this, std::placeholders::_1));
        joint_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "panda_joint_states", 10, std::bind(&MotionPlanner::jointCallback, this, std::placeholders::_1));
        trajectory_publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_trajectory", 10);
    }

private:
    
    void graspCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received grasp pose: position(%f, %f, %f), orientation(%f, %f, %f, %f)",
                    msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
                    msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
        // Implement motion planning logic here
        Eigen::Isometry3d T_world_grasp = Eigen::Isometry3d::Identity();
        Eigen::Vector3d pos_world(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        Eigen::Quaterniond ori_world(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
        T_world_grasp.translation() = pos_world;
        T_world_grasp.linear() = ori_world.toRotationMatrix();

        // Use the same hand offset compensation as the IK test.
        Eigen::Vector3d panda_base_pos = T_world_base_.translation();
        Eigen::Quaterniond panda_base_ori(T_world_base_.linear());
        // Pass the same HAND_OFFSET sign as in test_ik.cpp (HAND_OFFSET is negative there).
        Eigen::Isometry3d T_base_grasp = transformToPandaBase(
            pos_world,
            ori_world,
            panda_base_pos,
            panda_base_ori,
            HAND_OFFSET
        );

        RCLCPP_INFO(this->get_logger(), "Computed T_base_grasp: translation(%f, %f, %f)", 
                    T_base_grasp.translation().x(), T_base_grasp.translation().y(), T_base_grasp.translation().z());
        // Further processing, such as calling the IK solver, can be done here    

        // call the IK solver with the computed T_base_grasp and current joint states
        std::array<double, 7> q_actual = {
        0.0,
        1.0,
        0.0,
        -1.0,
        0.0,
        1.5,
        0.0
    };

        std::vector<std::array<double, 7>> q_solution;
        if (solvePandaIK(T_base_grasp, q_actual, q_solution))
        {
            RCLCPP_INFO(this->get_logger(), "IK solution found: [%f, %f, %f, %f, %f, %f, %f]", 
                        q_solution[0][0], q_solution[0][1], q_solution[0][2], q_solution[0][3], q_solution[0][4], q_solution[0][5], q_solution[0][6]);
            // Publish the joint trajectory
            trajectory_msgs::msg::JointTrajectory trajectory_msg;
        }
    }
    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // RCLCPP_INFO(this->get_logger(), "Received joint states: %lu joints", msg->name.size());
        // Implement joint state handling logic here
        current_joint_states_.fill(0.0);
        for (size_t i = 0; i < msg->name.size() && i < 7; ++i)
        {
            current_joint_states_[i] = msg->position[i];
        }
    }
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr grasp_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_subscriber_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_publisher_;

    Eigen::Isometry3d T_world_base_;
    std::array<double, 7> current_joint_states_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MotionPlanner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}