cmake_minimum_required(VERSION 3.8)
project(smart_city_gz_plugins)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)

add_library(model_world_pose_publisher SHARED
  src/model_world_pose_publisher.cpp
)

target_link_libraries(model_world_pose_publisher
  gz-sim8::gz-sim8
  gz-plugin2::gz-plugin2
)

ament_target_dependencies(model_world_pose_publisher
  rclcpp
  geometry_msgs
)

install(
  TARGETS model_world_pose_publisher
  LIBRARY DESTINATION lib
)

ament_package()