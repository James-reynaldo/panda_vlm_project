#include <array>
#include <cmath>
#include <iostream>

#include "panda_planner/Eigen/Dense"
#include "panda_planner/Eigen/Geometry"

#include "panda_planner/franka_ik_He.hpp"


// ============================================================
// Transform a pose from WORLD frame to PANDA BASE frame
// ============================================================

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


int main()
{
    // ============================================================
    // 1. GRASP POSE FROM TASK PLANNER
    //    Expressed in WORLD frame
    // ============================================================

    Eigen::Vector3d grasp_position_world(
        0.0,  0.05,   0.9
    );
    // Eigen::Vector3d grasp_position_world(
    //     0.0,
    //     0,
    //     1.12 
    // );

    // ROS quaternion is [x, y, z, w]
    //
    // Eigen::Quaterniond constructor is (w, x, y, z)

    // Eigen::Quaterniond grasp_orientation_world(
    //     6.123234e-17,  // w
    //     1.0,           // x
    //     0.0,           // y
    //     0.0            // z
    // );

    Eigen::Quaterniond grasp_orientation_world(
        0.5, -0.5,  0.5,  0.5       
    );
    // ============================================================
    // 2. PANDA BASE POSE FROM MUJOCO
    //    Expressed in WORLD frame
    // ============================================================

    Eigen::Vector3d panda_base_position_world(
        0.0,
        -0.65,
        0.912
    );

    Eigen::Quaterniond panda_base_orientation_world(
        0.70710678, 0 ,        0.        , 0.70710678
    );


    // ============================================================
    // 3. WORLD -> PANDA BASE
    // ============================================================
    constexpr double HAND_OFFSET = -0.107; // offset from IK to the grasp point

    Eigen::Isometry3d T_base_grasp =
        transformToPandaBase(
            grasp_position_world,
            grasp_orientation_world,
            panda_base_position_world,
            panda_base_orientation_world,
            HAND_OFFSET
        );
    


    // ============================================================
    // 4. PRINT TRANSFORMED POSE
    // ============================================================

    std::cout << "\n=== GRASP POSE IN PANDA BASE FRAME ===\n";

    std::cout
        << "Position: "
        << T_base_grasp.translation().transpose()
        << std::endl;

    std::cout
        << "Transformation matrix:\n"
        << T_base_grasp.matrix()
        << std::endl;


    // ============================================================
    // 5. CONVERT TO IK FORMAT
    // ============================================================

    std::array<double, 16> O_T_EE =
        toIKMatrix(T_base_grasp);
    
    printf("O_T_EE:\n");
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            printf("% .6f ", O_T_EE[j * 4 + i]);
        }
        printf("\n");
    }
    // std::array<double, 16> O_T_EE = {
    //     // column 1 (+X)
    //     1.0,
    //     0.0,
    //     0.0,
    //     0.0,

    //     // column 2 (+Y)
    //     0.0,
    //     -1.0,
    //     0.0,
    //     0.0,

    //     // column 3 (+Z) -> points downward
    //     0.0,
    //     0.0,
    //     -1.0,
    //     0.0,

    //     // column 4 = position
    //     0.5,
    //     0.0,
    //     0.1,
    //     1.0
    // };


    // ============================================================
    // 6. CURRENT PANDA CONFIGURATION
    // ============================================================

    std::array<double, 7> q_actual = {
        0.0,
        1.0,
        0.0,
        -1.0,
        0.0,
        1.5,
        0.0
    };


    for (double q7 = -3.0; q7 <= 3.0; q7 += 0.25)
    {
        auto solutions = franka_IK_EE(
            O_T_EE,
            q7,
            q_actual
        );

        for (size_t i = 0; i < solutions.size(); ++i)
        {
            if (!std::isnan(solutions[i][0]))
            {
                std::cout << "q7 = " << q7
                        << ", solution " << i << ": ";

                for (double q : solutions[i])
                    std::cout << q << " ";

                std::cout << std::endl;
            }
        }
    }

    return 0;
}