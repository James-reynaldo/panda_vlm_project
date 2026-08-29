cmake_minimum_required(VERSION 3.8)
project(panda_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
add_executable(motion_planner src/motion_planner.cpp)
target_include_directories(motion_planner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(motion_planner rclcpp geometry_msgs sensor_msgs trajectory_msgs)

add_executable(test_ik src/test_ik.cpp)
target_include_directories(test_ik PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
install(TARGETS motion_planner test_ik DESTINATION lib/${PROJECT_NAME})
ament_package()
