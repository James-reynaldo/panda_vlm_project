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
    std::array<double, 7>& q_solution)
{
    std::array<double, 16> O_T_EE =
        toIKMatrix(T_base_grasp);

    // Search over redundant joint q7
    for (double q7 = -3.0; q7 <= 3.0; q7 += 0.05)
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

            q_solution = solution;
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
            "joint_states", 10, std::bind(&MotionPlanner::jointCallback, this, std::placeholders::_1));
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
        T_world_grasp.translation() = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        T_world_grasp.linear() = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z).toRotationMatrix();

        Eigen::Isometry3d T_base_grasp = T_world_base_.inverse() * T_world_grasp;

        RCLCPP_INFO(this->get_logger(), "Computed T_base_grasp: translation(%f, %f, %f)", 
                    T_base_grasp.translation().x(), T_base_grasp.translation().y(), T_base_grasp.translation().z());
        // Further processing, such as calling the IK solver, can be done here    
    }
    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received joint states: %lu joints", msg->name.size());
        // Implement joint state handling logic here
    }
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr grasp_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_subscriber_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_publisher_;

    Eigen::Isometry3d T_world_base_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MotionPlanner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}