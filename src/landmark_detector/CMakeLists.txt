cmake_minimum_required(VERSION 3.5)
project(landmark_detector)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages in the 'msg' folder
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Landmark.msg"
  "msg/LandmarkArray.msg"
  DEPENDENCIES std_msgs
)

# Since it's a Python package, no custom libraries to build.
ament_package()
