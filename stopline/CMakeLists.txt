cmake_minimum_required(VERSION 2.8.3)
project(stopline)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport 
  roscpp
  std_msgs
  rospy
  tf
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(stopline src/stopline.cpp)

target_link_libraries(stopline
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

