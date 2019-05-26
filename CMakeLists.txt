cmake_minimum_required(VERSION 2.8.3)

# package name 
project(lanes)

set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CFLAGE "-w")

# Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	image_transport 
	roscpp
	std_msgs
	rospy
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/Tutorials.cfg
  #...
)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Armadillo REQUIRED)

catkin_package()

include_directories(
	include
	${catkin_INCLUDE_DIRS}
  	${OpenCV_INCLUDE_DIRS}
  	${CUDA_INCLUDE_DIRS}
  	${ARMADILLO_INCLUDE_DIRS}
)

add_executable(lanes_ransac src/laneDetector_ransac.cpp)
add_executable(lanes src/laneDetector.cpp)

target_link_libraries(lanes 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
)

target_link_libraries(lanes_ransac 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
)

add_dependencies(lanes_ransac ${PROJECT_NAME}_gencfg)