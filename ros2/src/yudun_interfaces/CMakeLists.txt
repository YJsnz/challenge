cmake_minimum_required(VERSION 3.8)
project(yudun_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection2D.msg" "msg/Detection3D.msg" "msg/DetectionArray.msg"
  "msg/FusedTrack.msg" "msg/FusedTrackArray.msg" "msg/RiskEvent.msg" "msg/RiskEventArray.msg"
  DEPENDENCIES std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
