#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>   
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <array>
#include <chrono>
#include <future>
#include <string>
#include <vector>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/PathSimplifier.h>

#include "panda_interfaces/srv/collision_check.hpp"

#include "panda_planner/Eigen/Dense"
#include "panda_planner/Eigen/Geometry"

#include "panda_planner/franka_ik_He.hpp"
#include "panda_interfaces/srv/plan_motion.hpp"

constexpr double HAND_OFFSET = -0.107;
constexpr double OMPL_PLANNING_TIME = 10.0;

static constexpr double PANDA_BASE_X = 0.0;
static constexpr double PANDA_BASE_Y = -0.65;
static constexpr double PANDA_BASE_Z = 0.912;

static constexpr double PANDA_BASE_QW = 0.70710678;
static constexpr double PANDA_BASE_QX = 0.0;
static constexpr double PANDA_BASE_QY = 0.0;
static constexpr double PANDA_BASE_QZ = 0.70710678;

static constexpr std::array<double, 7> PANDA_JOINT_LIMITS_LOW = {
    -2.7473, -1.6728, -2.7473, -2.9218, -2.7473, -0.0775, -2.7473
};

static constexpr std::array<double, 7> PANDA_JOINT_LIMITS_HIGH = {
    2.7473, 1.6728, 2.7473, -0.2198, 2.7473, 3.6025, 2.7473
};

static constexpr std::array<const char*, 7> PANDA_JOINT_NAMES = {
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"
};

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

bool isWithinJointLimits(const std::array<double, 7>& q)
{
    for (std::size_t i = 0; i < 7; ++i)
    {
        if (q[i] < PANDA_JOINT_LIMITS_LOW[i] || q[i] > PANDA_JOINT_LIMITS_HIGH[i])
        {
            return false;
        }
    }
    return true;
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

            if (!isWithinJointLimits(solution))
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

        collision_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        joint_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "panda_joint_states", 10, std::bind(&MotionPlanner::jointCallback, this, std::placeholders::_1));
        trajectory_publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_trajectory", 10);
        collision_checker_client_ = this->create_client<panda_interfaces::srv::CollisionCheck>(
            "/collision_check",
            rmw_qos_profile_services_default,
            collision_callback_group_);
        plan_motion_server_ = this->create_service<panda_interfaces::srv::PlanMotion>(
            "/plan_motion",
            std::bind(&MotionPlanner::handlePlanMotion, this, std::placeholders::_1, std::placeholders::_2));

        current_joint_states_.fill(0.0);
        current_joint_names_.clear();
        for (const char* joint_name : PANDA_JOINT_NAMES)
        {
            current_joint_names_.emplace_back(joint_name);
        }

        collision_service_available_ = collision_checker_client_->wait_for_service(std::chrono::seconds(5));
        if (!collision_service_available_) {
            RCLCPP_ERROR(this->get_logger(), "Collision checker service '/collision_check' is unavailable during initialization.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Collision checker service '/collision_check' available.");
        }
    }

    bool checkCollision(const std::array<double, 7>& q)
    {
        if (!collision_checker_client_) {
            RCLCPP_ERROR(this->get_logger(), "Collision checker service client is unavailable.");
            return false;
        }

        if (!collision_service_available_) {
            RCLCPP_ERROR(this->get_logger(), "Collision checker service '/collision_check' was not available at startup.");
            return false;
        }

        auto request = std::make_shared<panda_interfaces::srv::CollisionCheck::Request>();
        request->joint_positions.assign(q.begin(), q.end());

        auto future = collision_checker_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Collision checker service call timed out for joint configuration.");
            return false;
        }

        try {
            auto response = future.get();
            if (!response->collision_free) {
                RCLCPP_WARN(this->get_logger(), "Collision detected for q = [%f, %f, %f, %f, %f, %f, %f]",
                            q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
                return false;
            }

            return true;
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "Collision checker service call failed: %s", ex.what());
            return false;
        }
    }

    bool planJointTrajectory(const std::array<double, 7>& q_goal, trajectory_msgs::msg::JointTrajectory& trajectory_msg)
    {
        if (!isWithinJointLimits(q_goal))
        {
            RCLCPP_ERROR(this->get_logger(), "Goal joint configuration is outside Panda limits and cannot be used by OMPL.");
            return false;
        }

        auto space = std::make_shared<ompl::base::RealVectorStateSpace>(7);
        ompl::base::RealVectorBounds bounds(7);
        for (std::size_t i = 0; i < 7; ++i)
        {
            bounds.setLow(i, PANDA_JOINT_LIMITS_LOW[i]);
            bounds.setHigh(i, PANDA_JOINT_LIMITS_HIGH[i]);
        }
        space->setBounds(bounds);

        ompl::geometric::SimpleSetup ss(space);
        ss.setStateValidityChecker([this](const ompl::base::State* state) {
            const auto* rvs = state->as<ompl::base::RealVectorStateSpace::StateType>();
            std::array<double, 7> q{};
            for (std::size_t i = 0; i < 7; ++i)
            {
                q[i] = rvs->values[i];
            }
            return this->checkCollision(q);
        });

        ompl::base::ScopedState<ompl::base::RealVectorStateSpace> start(space);
        for (std::size_t i = 0; i < 7; ++i)
        {
            start[i] = current_joint_states_[i];
        }

        ompl::base::ScopedState<ompl::base::RealVectorStateSpace> goal(space);
        for (std::size_t i = 0; i < 7; ++i)
        {
            goal[i] = q_goal[i];
        }

        ss.setStartAndGoalStates(start, goal);
        ss.setPlanner(std::make_shared<ompl::geometric::RRTConnect>(ss.getSpaceInformation()));

        const ompl::base::PlannerStatus status = ss.solve(OMPL_PLANNING_TIME);
        if (!status)
        {
            RCLCPP_ERROR(this->get_logger(), "OMPL planning failed to initialize for the requested goal.");
            return false;
        }

        if (!ss.haveSolutionPath())
        {
            RCLCPP_WARN(this->get_logger(), "No collision-free OMPL path found within %.1f seconds.", OMPL_PLANNING_TIME);
            return false;
        }

        ompl::geometric::PathGeometric path = ss.getSolutionPath();
        path.interpolate();

        // Log path statistics before simplification
        size_t states_before = path.getStateCount();
        RCLCPP_INFO(this->get_logger(), "Initial path has %zu states.", states_before);

        // Simplify the path using OMPL's PathSimplifier
        ompl::geometric::PathSimplifier ps(ss.getSpaceInformation());
        ps.simplify(path, OMPL_PLANNING_TIME * 0.1);  // Use 10% of planning time for simplification

        // Log path statistics after simplification
        size_t states_after = path.getStateCount();
        double path_length = path.length();
        RCLCPP_INFO(this->get_logger(), "Simplified path has %zu states (reduced by %zu). Path length: %.4f",
                    states_after, states_before - states_after, path_length);

        trajectory_msg.header.stamp = this->now();
        trajectory_msg.header.frame_id = "world";
        trajectory_msg.joint_names = current_joint_names_;
        trajectory_msg.points.resize(path.getStateCount());

        const double dt = path.getStateCount() > 1 ? 0.1 : 0.0;
        for (std::size_t i = 0; i < path.getStateCount(); ++i)
        {
            const auto* state = path.getState(i)->as<ompl::base::RealVectorStateSpace::StateType>();
            trajectory_msgs::msg::JointTrajectoryPoint point;
            point.positions.resize(7);
            point.velocities.resize(7, 0.0);
            point.accelerations.resize(7, 0.0);

            for (std::size_t j = 0; j < 7; ++j)
            {
                point.positions[j] = state->values[j];
            }

            point.time_from_start = rclcpp::Duration::from_seconds(static_cast<double>(i) * dt);
            trajectory_msg.points[i] = point;
        }

        RCLCPP_INFO(this->get_logger(), "OMPL found a collision-free path with %zu points.", trajectory_msg.points.size());
        return true;
    }

private:
    void handlePlanMotion(
        const std::shared_ptr<panda_interfaces::srv::PlanMotion::Request> request,
        std::shared_ptr<panda_interfaces::srv::PlanMotion::Response> response)
    {
        if (planning_in_progress_.load(std::memory_order_acquire))
        {
            response->success = false;
            response->message = "Motion planning is already in progress.";
            response->trajectory = trajectory_msgs::msg::JointTrajectory();
            RCLCPP_WARN(this->get_logger(), "Ignoring /plan_motion request while a plan is already in progress.");
            return;
        }

        planning_in_progress_.store(true, std::memory_order_release);

        try
        {
            const auto& grasp_pose = request->grasp_pose;
            RCLCPP_INFO(this->get_logger(), "Received /plan_motion request: position(%f, %f, %f), orientation(%f, %f, %f, %f)",
                        grasp_pose.pose.position.x, grasp_pose.pose.position.y, grasp_pose.pose.position.z,
                        grasp_pose.pose.orientation.x, grasp_pose.pose.orientation.y, grasp_pose.pose.orientation.z, grasp_pose.pose.orientation.w);

            Eigen::Vector3d pos_world(grasp_pose.pose.position.x, grasp_pose.pose.position.y, grasp_pose.pose.position.z);
            Eigen::Quaterniond ori_world(grasp_pose.pose.orientation.x, grasp_pose.pose.orientation.y, grasp_pose.pose.orientation.z, grasp_pose.pose.orientation.w);

            Eigen::Isometry3d T_base_grasp = transformToPandaBase(
                pos_world,
                ori_world,
                T_world_base_.translation(),
                Eigen::Quaterniond(T_world_base_.linear()),
                HAND_OFFSET
            );

            std::vector<std::array<double, 7>> q_solution;
            if (!solvePandaIK(T_base_grasp, current_joint_states_, q_solution))
            {
                response->success = false;
                response->message = "No IK solution found for the requested grasp pose.";
                response->trajectory = trajectory_msgs::msg::JointTrajectory();
                return;
            }

            const auto& candidate = q_solution[0];
            RCLCPP_INFO(this->get_logger(), "IK solution found: [%f, %f, %f, %f, %f, %f, %f]",
                        candidate[0], candidate[1], candidate[2], candidate[3], candidate[4], candidate[5], candidate[6]);

            if (!checkCollision(candidate))
            {
                response->success = false;
                response->message = "Discarding IK solution because it is in collision or the service check failed.";
                response->trajectory = trajectory_msgs::msg::JointTrajectory();
                return;
            }

            trajectory_msgs::msg::JointTrajectory trajectory_msg;
            if (!planJointTrajectory(candidate, trajectory_msg))
            {
                response->success = false;
                response->message = "Planning failed for the IK goal within the configured time budget.";
                response->trajectory = trajectory_msgs::msg::JointTrajectory();
                return;
            }

            trajectory_publisher_->publish(trajectory_msg);
            response->success = true;
            response->message = "Motion plan generated successfully.";
            response->trajectory = trajectory_msg;
            RCLCPP_INFO(this->get_logger(), "Published joint trajectory on 'joint_trajectory' and returned it via /plan_motion.");
        }
        catch (const std::exception& ex)
        {
            response->success = false;
            response->message = std::string("Planning service failed: ") + ex.what();
            response->trajectory = trajectory_msgs::msg::JointTrajectory();
            RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
        }

        planning_in_progress_.store(false, std::memory_order_release);
    }

    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        current_joint_states_.fill(0.0);
        current_joint_names_.clear();

        if (!msg->name.empty())
        {
            current_joint_names_.reserve(msg->name.size());
            for (size_t i = 0; i < msg->name.size() && i < 7; ++i)
            {
                current_joint_names_.push_back(msg->name[i]);
                current_joint_states_[i] = msg->position[i];
            }
        }

        if (current_joint_names_.size() < 7)
        {
            current_joint_names_.clear();
            for (const char* joint_name : PANDA_JOINT_NAMES)
            {
                current_joint_names_.emplace_back(joint_name);
            }
        }
        else if (current_joint_names_.size() > 7)
        {
            current_joint_names_.resize(7);
        }
    }
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_subscriber_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_publisher_;
    rclcpp::Service<panda_interfaces::srv::PlanMotion>::SharedPtr plan_motion_server_;
    rclcpp::CallbackGroup::SharedPtr collision_callback_group_;
    rclcpp::Client<panda_interfaces::srv::CollisionCheck>::SharedPtr collision_checker_client_;
    bool collision_service_available_{false};
    std::atomic_bool planning_in_progress_{false};

    Eigen::Isometry3d T_world_base_;
    std::array<double, 7> current_joint_states_{};
    std::vector<std::string> current_joint_names_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MotionPlanner>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}