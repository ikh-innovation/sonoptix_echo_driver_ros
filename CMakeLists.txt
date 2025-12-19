cmake_minimum_required(VERSION 3.0.2)
project(sonoptix_echo_driver_ros)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_srvs
  cv_bridge
  sensor_msgs
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/SonoptixEcho.cfg
)

# TODO: Add setup.py and uncomment
# catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_srvs
    cv_bridge
    sensor_msgs
    dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

