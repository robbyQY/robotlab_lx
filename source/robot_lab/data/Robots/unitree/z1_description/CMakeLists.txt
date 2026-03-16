cmake_minimum_required(VERSION 3.0.2)
project(z1_description)

find_package(catkin REQUIRED COMPONENTS
  urdf
  xacro
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_custom_target(
    z1_generate_urdf ALL
    COMMAND rosrun xacro xacro --inorder ${PROJECT_SOURCE_DIR}/xacro/robot.xacro -o ${PROJECT_SOURCE_DIR}/urdf/z1.urdf
    DEPENDS ${PROJECT_SOURCE_DIR}/xacro/const.xacro
    COMMENT "Generating URDF from XACRO"
)

# 确保urdf目录存在
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/xacro)